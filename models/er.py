# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
import numpy as np
import ipdb


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0
        self.n_iters = 0
        self.total_iters = 1000

    def end_task(self, dataset):
        self.task += 1
        self.n_iters = 0

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        return loss.item()
    
    # def observe(self, inputs, labels, not_aug_inputs):
    #     real_batch_size = inputs.shape[0]
    #     self.opt.zero_grad()
    #     if not self.buffer.is_empty():
    #         if self.task:
    #             p = min(self.n_iters/self.total_iters, 1)
    #             if p > 0:
    #                 mask = torch.bernoulli(torch.ones_like(labels)*p, p)
    #             else:
    #                 mask = torch.zeros_like(labels)
    #             buf_inputs, buf_labels = self.buffer.get_data(
    #                 self.args.minibatch_size+len(inputs)-mask.long().sum().item(), transform=self.transform)
    #             inputs = torch.cat((inputs[mask==1], buf_inputs))
    #             labels = torch.cat((labels[mask==1], buf_labels))
    #         else:
    #             buf_inputs, buf_labels = self.buffer.get_data(
    #                 self.args.minibatch_size, transform=self.transform)
    #             inputs = torch.cat((inputs, buf_inputs))
    #             labels = torch.cat((labels, buf_labels))

    #     outputs = self.net(inputs)
    #     loss = self.loss(outputs, labels)

    #     loss.backward()
    #     self.opt.step()
    #     self.buffer.add_data(examples=not_aug_inputs,
    #                          labels=labels[:real_batch_size])
    #     self.n_iters += 1
    #     return loss.item()

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):

        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index,:]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
