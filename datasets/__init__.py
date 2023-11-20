# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_cifar100_10 import SequentialCIFAR100_10
from datasets.seq_cifar100_4 import SequentialCIFAR100_4
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialCIFAR100_10.NAME: SequentialCIFAR100_10,
    SequentialCIFAR100_4.NAME: SequentialCIFAR100_4,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
}

def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)

