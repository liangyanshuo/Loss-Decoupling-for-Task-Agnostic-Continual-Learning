# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

best_args = {
    'seq-cifar10': {'sgd': {-1: {'lr': 0.1,
                                 'batch_size': 32,
                                 'n_epochs': 50}},
                    'icarl': {500: {'lr': 0.1,
                                    'minibatch_size': 0,
                                    'softmax_temp': 2.0,
                                    'wd_reg': 0.00001,
                                    'batch_size': 32,
                                    'n_epochs': 50},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 0,
                                     'softmax_temp': 2.0,
                                     'wd_reg': 0.00001,
                                     'batch_size': 32,
                                     'n_epochs': 50}},
                    'er': {500: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 50},
                           5120: {'lr': 0.1,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 50}},
                    'lode_er': {500: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'rho': 0.1,
                                 'n_epochs': 50},
                           5120: {'lr': 0.1,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'rho': 0.1,
                                  'n_epochs': 50}},
                    'derpp': {500: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.2,
                                    'beta': 0.5,
                                    'batch_size': 32,
                                    'n_epochs': 50},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'alpha': 0.1,
                                     'beta': 1.0,
                                     'batch_size': 32,
                                     'n_epochs': 50}},
                    'lode_derpp': {500: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.2,
                                    'beta': 0.5,
                                    'batch_size': 32,
                                    'rho': 0.05,
                                    'n_epochs': 50,
                                    'C': 1.0,},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'alpha': 0.1,
                                     'beta': 1.0,
                                     'batch_size': 32,
                                     'rho': 0.1,
                                     'n_epochs': 50,
                                     'C': 1.0}},
                    'esmer': {
                        500: {
                            'reg_weight': 0.15,
                            'ema_model_alpha': 0.999,
                            'ema_model_update_freq': 0.1,
                            'loss_margin': 1.2,
                            'loss_alpha': 0.99,
                            'lr': 0.03,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'n_epochs': 50,
                            'task_warmup': 1,
                            'std_margin': 1,
                        },
                        5120: {
                            'reg_weight': 0.15,
                            'ema_model_alpha': 0.999,
                            'ema_model_update_freq': 0.1,
                            'loss_margin': 1.2,
                            'loss_alpha': 0.99,
                            'lr': 0.03,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'n_epochs': 50,
                            'task_warmup': 1,
                            'std_margin': 1,}},
                    'lode_esmer': {
                        500: {
                            'reg_weight': 0.15,
                            'ema_model_alpha': 0.998,
                            'ema_model_update_freq': 0.1,
                            'loss_margin': 1.2,
                            'loss_alpha': 0.99,
                            'lr': 0.03,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'rho': 0.1,
                            'C': 1.0,
                            'n_epochs': 50,
                            'task_warmup': 1,
                            'std_margin': 1,
                        },
                        5120: {
                            'reg_weight': 0.15,
                            'ema_model_alpha': 0.998,
                            'ema_model_update_freq': 0.1,
                            'loss_margin': 1.2,
                            'loss_alpha': 0.99,
                            'lr': 0.03,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'rho': 0.1,
                            'C': 1.0,
                            'n_epochs': 50,
                            'task_warmup': 1,
                            'std_margin': 1,}},
                                     },
    'seq-cifar100': {'sgd': {-1: {'lr': 0.1,
                                 'batch_size': 32,
                                 'n_epochs': 50}},
                    'icarl': {500: {'lr': 0.1,
                                    'minibatch_size': 0,
                                    'softmax_temp': 2.0,
                                    'wd_reg': 0.00001,
                                    'batch_size': 32,
                                    'n_epochs': 50},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 0,
                                     'softmax_temp': 2.0,
                                     'wd_reg': 0.00001,
                                     'batch_size': 32,
                                     'n_epochs': 50}},
                    'er': {500: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 50},
                           5120: {'lr': 0.1,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 50}},
                    'lode_er': {500: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'rho': 0.05,
                                 'n_epochs': 50},
                           5120: {'lr': 0.1,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'rho': 0.01,
                                  'n_epochs': 50}},
                    'derpp': {500: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.2,
                                    'beta': 0.5,
                                    'batch_size': 32,
                                    'n_epochs': 50},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'alpha': 0.2,
                                     'beta': 0.5,
                                     'batch_size': 32,
                                     'n_epochs': 50}},
                    'lode_derpp': {500: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.2,
                                    'beta': 0.5,
                                    'batch_size': 32,
                                    'rho': 0.1,
                                    'n_epochs': 50,
                                    'C': 1.0},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'alpha': 0.2,
                                     'beta': 0.2,
                                     'batch_size': 32,  
                                     'rho': 0.5,
                                     'n_epochs': 50,
                                     'C': 1.0}},
                    'esmer': {500: {
                                     'reg_weight': 0.15,
                                     'ema_model_alpha': 0.999,
                                     'ema_model_update_freq': 0.07,
                                     'loss_margin': 1.0,
                                     'loss_alpha': 0.99,
                                     'lr': 0.03,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 50,
                                     'task_warmup': 1,
                                     'std_margin': 1,},
                                5120: {
                                        'reg_weight': 0.15,
                                        'ema_model_alpha': 0.999,
                                        'ema_model_update_freq': 0.07,
                                        'loss_margin': 1.0,
                                        'loss_alpha': 0.99,
                                        'lr': 0.03,
                                        'minibatch_size': 32,
                                        'batch_size': 32,
                                        'n_epochs': 50,
                                        'task_warmup': 1,
                                        'std_margin': 1,}},
                    'lode_esmer': {500: {
                                     'reg_weight': 0.15,
                                     'ema_model_alpha': 0.998,
                                     'ema_model_update_freq': 0.07,
                                     'loss_margin': 1.0,
                                     'loss_alpha': 0.99,
                                     'lr': 0.03,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'rho': 0.05,
                                     'C': 1.0,
                                     'n_epochs': 50,
                                     'task_warmup': 1,
                                     'std_margin': 1,},
                                5120: {
                                        'reg_weight': 0.15,
                                        'ema_model_alpha': 0.998,
                                        'ema_model_update_freq': 0.07,
                                        'loss_margin': 1.0,
                                        'loss_alpha': 0.99,
                                        'lr': 0.03,
                                        'minibatch_size': 32,
                                        'batch_size': 32,
                                        'rho': 0.5,
                                        'C': 1.0,
                                        'n_epochs': 50,
                                        'task_warmup': 1,
                                        'std_margin': 1,}},},
    'seq-tinyimg': {'sgd': {-1: {'lr': 0.03,
                                 'batch_size': 32,
                                 'n_epochs': 100}},
                    'icarl': {500: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'softmax_temp': 2.0,
                                    'wd_reg': 0.00001,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'softmax_temp': 2.0,
                                     'wd_reg': 0.00001,
                                     'batch_size': 32,
                                     'n_epochs': 100}},
                    'er': {500: {'lr': 0.03,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 100},
                           5120: {'lr': 0.1,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 100}},
                    'lode_er': {500: {'lr': 0.03,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'rho': 0.05,
                                 'n_epochs': 100},
                           5120: {'lr': 0.1,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'rho': 0.05,
                                  'n_epochs': 100}},
                    'derpp': {500: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.2,
                                    'beta': 0.5,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'alpha': 0.1,
                                     'beta': 0.5,
                                     'batch_size': 32,
                                     'n_epochs': 100}},
                    'lode_derpp': {500: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.2,
                                    'beta': 0.5,
                                    'batch_size': 32,
                                    'rho': 0.1,
                                    'n_epochs': 25,
                                    'C': 1.0},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'alpha': 0.1,
                                     'beta': 0.5,
                                     'batch_size': 32,
                                     'rho': 0.1,
                                     'n_epochs': 25,
                                     'C': 2.0}},
                    'esmer': {
                        500: {
                            'reg_weight': 0.15,
                            'ema_model_alpha': 0.999,
                            'ema_model_update_freq': 0.07,
                            'loss_margin': 1.0,
                            'loss_alpha': 0.99,
                            'lr': 0.03,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'n_epochs': 50,
                            'task_warmup': 1,
                            'std_margin': 1,},
                        5120: {
                            'reg_weight': 0.15,
                            'ema_model_alpha': 0.999,
                            'ema_model_update_freq': 0.07,
                            'loss_margin': 1.0,
                            'loss_alpha': 0.99,
                            'lr': 0.03,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'n_epochs': 50,
                            'task_warmup': 1,
                            'std_margin': 1,}},
                    'lode_esmer': {
                        500: {
                            'reg_weight': 0.15,
                            'ema_model_alpha': 0.998,
                            'ema_model_update_freq': 0.07,
                            'loss_margin': 1.0,
                            'loss_alpha': 0.99,
                            'lr': 0.03,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'rho': 0.05,
                            'C': 1.0,
                            'n_epochs': 50,
                            'task_warmup': 1,
                            'std_margin': 1,},
                        5120: {
                            'reg_weight': 0.15,
                            'ema_model_alpha': 0.998,
                            'ema_model_update_freq': 0.07,
                            'loss_margin': 1.0,
                            'loss_alpha': 0.99,
                            'lr': 0.03,
                            'minibatch_size': 32,
                            'batch_size': 32,
                            'rho': 0.1,
                            'C': 1.0,
                            'n_epochs': 50,
                            'task_warmup': 1,
                            'std_margin': 1,}},
                            },
}
