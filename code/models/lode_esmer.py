import torch
from utils.buffer_esmer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from copy import deepcopy
from torch import nn
import os
import torch.nn.functional as F
import ipdb

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Complementary Learning Systems Based Experience Replay')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    # Consistency Regularization Weight
    parser.add_argument('--reg_weight', type=float, default=0.1)
    # Stable Model parameters
    parser.add_argument('--ema_model_update_freq', type=float, default=0.70)
    parser.add_argument('--ema_model_alpha', type=float, default=0.999)
    # Training modes
    parser.add_argument('--save_interim', type=int, default=0)
    # Sample Selection
    parser.add_argument('--loss_margin', type=float, default=1)
    parser.add_argument('--loss_alpha', type=float, default=0.99)
    parser.add_argument('--std_margin', type=float, default=1)
    parser.add_argument('--task_warmup', type=int, default=1)

    parser.add_argument('--rho', type=float, default=0.1)

    parser.add_argument('--epsilon', type=float, required=True,
                        help='Penalty weight.')
    return parser

# 没用改进的loss
def logmeanexp_previous(x, classes1, classes2, dim=None):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim)
    x_max = x_max.detach()
    old_pre = torch.logsumexp(x[:, classes1], dim=1)
    new_pre = torch.logsumexp(x[:, classes2], dim=1)
    pre = torch.stack([old_pre, new_pre], dim=-1)
    return pre

# =============================================================================
# Mean-ER
# =============================================================================
class LODEESMER(ContinualModel):
    NAME = 'lodeesmer'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(LODEESMER, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        # Initialize EMA model
        self.ema_model = deepcopy(self.net).to(self.device)
        # set regularization weight
        self.reg_weight = args.reg_weight
        # set parameters for ema model
        self.ema_model_update_freq = args.ema_model_update_freq
        self.ema_model_alpha = args.ema_model_alpha
        # Set loss functions
        self.consistency_loss = nn.MSELoss(reduction='none')
        self.task_loss = nn.CrossEntropyLoss(reduction='none')

        self.current_task = 0
        self.global_step = 0
        self.task_iter = 0

        # Running estimates
        self.loss_running_sum = 0
        self.loss_running_mean = 0
        self.loss_running_std = 0
        self.n_samples_seen = 0
        self.seen_so_far = torch.LongTensor(size=(0,)).to(self.device)

        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.n_classes_per_task = get_dataset(args).N_CLASSES_PER_TASK
        
        self.avg_inter_loss = 0
        self.avg_intra_loss = 0
        
    def forward(self, x):
        mode = self.ema_model.training
        self.ema_model.eval()
        output = self.ema_model(x)
        self.ema_model.train(mode)
        return output

    def inter_cls(self, logits, y, classes1, classes2):
        inter_logits = logmeanexp_previous(logits, classes1, classes2, dim=-1)
        inter_y = torch.ones_like(y)
        return F.cross_entropy(inter_logits, inter_y, reduction='none')

    def intra_cls(self, logits, y, classes):
        mask = torch.zeros_like(logits)
        mask[:, classes] = 1
        logits1 = logits - (1 - mask) * 1e9
        # ipdb.set_trace()
        return F.cross_entropy(logits1, y, reduction='none')

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        self.net.train()
        self.ema_model.train()

        loss = 0

        # =====================================================================
        # Apply Selective Cross Entropy loss
        # =====================================================================
        ema_out = self.ema_model(inputs)
        # ema_model_loss = self.task_loss(ema_out, labels)

        out = self.net(inputs)

        # =====================================================================
        # Apply Buffer loss
        # =====================================================================
        if not self.buffer.is_empty():
            if self.current_task:
                old_classes = torch.arange(self.current_task*self.n_classes_per_task)
                new_classes = torch.arange(self.current_task*self.n_classes_per_task, self.num_classes)
                assert len(old_classes) + len(new_classes) == self.num_classes

                new_inter_cls = self.inter_cls(out, labels, old_classes, new_classes)
                new_intra_cls = self.intra_cls(out, labels, new_classes)

                ema_new_inter_cls = self.inter_cls(ema_out, labels, old_classes, new_classes)
                ema_new_intra_cls = self.intra_cls(ema_out, labels, new_classes)

                if self.avg_inter_loss:
                    inter_alpha = torch.clamp(self.avg_inter_loss/(new_inter_cls.detach()+1e-5),max=1,min=1-self.args.epsilon)
                    intra_alpha = torch.clamp(self.avg_intra_loss/(new_intra_cls.detach()+1e-5),max=1,min=1-self.args.epsilon)
                else:
                    inter_alpha = torch.ones_like(ema_new_inter_cls)
                    intra_alpha = torch.ones_like(ema_new_intra_cls)
                # inter_alpha = inter_alpha/inter_alpha.sum()
                # intra_alpha = intra_alpha/intra_alpha.sum()
                loss = self.args.rho/self.current_task*(inter_alpha*new_inter_cls).mean() + (intra_alpha*new_intra_cls).mean()
            else:
                task_loss = self.task_loss(out, labels)
                ce_loss = task_loss.mean()
                loss += ce_loss

            buf_data = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

            buf_inputs, buf_labels = buf_data

            ema_model_logits = self.ema_model(buf_inputs)
            buf_out = self.net(buf_inputs)

            l_buf_cons = torch.mean(self.consistency_loss(buf_out, ema_model_logits.detach()))
            l_buf_ce = torch.mean(self.task_loss(buf_out, buf_labels))

            l_buf = self.args.reg_weight * l_buf_cons + l_buf_ce
            loss += l_buf
        else:
            task_loss = self.task_loss(out, labels)
            ce_loss = task_loss.mean()
            loss += ce_loss

        loss.backward()
        self.opt.step()

        # # if not self.warmup_phase: ipdb.set_trace()
        # if labels[~ignore_mask].any() and not self.warmup_phase:
        #     if hasattr(self, 'is_noise'):
        #         self.buffer.add_data(
        #             examples=not_aug_inputs[~ignore_mask],
        #             labels=labels[~ignore_mask],
        #             timestamps=torch.ones_like(labels[~ignore_mask]) * self.current_task,
        #             is_noise=self.is_noise[~ignore_mask]
        #         )
        #     else:
        #         self.buffer.add_data(
        #             examples=not_aug_inputs[~ignore_mask],
        #             labels=labels[~ignore_mask],
        #             timestamps=torch.ones_like(labels[~ignore_mask]) * self.current_task,
        #         )
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
        )

        # Update the ema model
        self.global_step += 1
        self.task_iter += 1
        if torch.rand(1) < self.ema_model_update_freq:
            self.update_ema_model_variables()

        # loss_mean, loss_std = ema_model_loss.mean(), ema_model_loss.std()
        # ignore_mask = ema_model_loss > (loss_mean + (self.args.std_margin * loss_std))
        # ema_model_loss = ema_model_loss[~ignore_mask]

        if not self.warmup_phase and self.current_task:
            self.update_running_loss_ema(ema_new_inter_cls.detach(), ema_new_intra_cls.detach())

        return loss.item()

    def update_ema_model_variables(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_model_alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.net.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def update_running_loss_ema(self, batch_inter, batch_intra):
        alpha = min(1 - 1 / (self.global_step + 1), 0.99)
        self.avg_inter_loss = alpha * self.avg_inter_loss + (1 - alpha) * batch_inter.mean()
        self.avg_intra_loss = alpha * self.avg_intra_loss + (1 - alpha) * batch_intra.mean()

    def end_task(self, dataset) -> None:
        self.current_task += 1
        self.task_iter = 0

    def begin_task(self, dataset):
        if self.args.task_warmup > 0:
            self.warmup_phase = 1
            print('Enabling Warmup phase')
        else:
            self.warmup_phase = 0

    def end_epoch(self, epoch, dataset) -> None:
        if (epoch >= self.args.task_warmup) and self.warmup_phase:
            self.warmup_phase = 0
            print('Disable Warmup phase')
