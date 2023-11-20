# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from datasets import get_dataset
from utils.args import *
import torch
import ipdb

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

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--rho', type=float, required=True,
                        help='Penalty weight.')
    return parser


class LODEDerpp(ContinualModel):
    NAME = 'lodederpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(LODEDerpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.n_classes_per_task = get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

    def end_task(self, dataset):
        self.task += 1

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
        outputs = self.net(inputs)

        if not self.buffer.is_empty():
            if self.task > 0:
                old_classes = torch.arange(self.task*self.n_classes_per_task)
                new_classes = torch.arange(self.task*self.n_classes_per_task, self.num_classes)
                assert len(old_classes) + len(new_classes) == self.num_classes
                buf_inputs, buf_labels, _, _ = self.buffer.get_data_exp(
                    self.args.minibatch_size, transform=self.transform)
                buf_outputs = self.net(buf_inputs)

                # all_output = torch.cat([outputs, buf_outputs])
                # all_labels = torch.cat([labels, buf_labels])
                # old_output = all_output[all_labels < len(old_classes)]
                # old_labels = all_labels[all_labels < len(old_classes)]
                # new_output = all_output[all_labels >= len(old_classes)]
                # new_labels = all_labels[all_labels >= len(old_classes)]
                # new_inter_cls = self.inter_cls(new_output, new_labels, old_classes, new_classes)
                # new_intra_cls = self.intra_cls(new_output, new_labels, new_classes)
                # loss = 1./self.task*new_inter_cls.mean() + new_intra_cls.mean() + self.args.alpha * F.cross_entropy(old_output, old_labels)

                new_inter_cls = self.inter_cls(outputs, labels, old_classes, new_classes)
                new_intra_cls = self.intra_cls(outputs, labels, new_classes)
                loss = self.args.rho/self.task*new_inter_cls.mean() + new_intra_cls.mean() + self.args.beta * F.cross_entropy(buf_outputs, buf_labels)
            else:
                buf_inputs, buf_labels, _, _ = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                buf_outputs = self.net(buf_inputs)
                loss = self.loss(outputs, labels) + self.args.beta * self.loss(buf_outputs, buf_labels)
            # loss /= 2.

            buf_inputs, _, buf_logits, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
        else:
            loss = self.loss(outputs, labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data, task_labels=torch.ones_like(labels)*self.task)

        return loss.item()
