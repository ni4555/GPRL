#!/usr/bin/python
# -*- coding:utf-8 -*-
import random
import os
import logging
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn

# import utlis.transfer_generator as tg


def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args_MAML():

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    parser.add_argument('--num_ways', type=int, default=5,
                        help='Number of classes per task (C in "C-way").')# C
    parser.add_argument('--num_shots', type=int, default=1,
                        help='Number of examples per class (K in "K-shot").')# K
    parser.add_argument('--num_querys', type=int, default=15,
                        help='Number of classes per task (M in "M-query").')# M
    parser.add_argument('--meta_lr', type=float, default=0.003,
                        help='Meta learning rate for MAML.')
    parser.add_argument('--update_lr', type=float, default=0.4,
                        help='Step size for the gradient step for adaptation.')
    parser.add_argument('--tasks_batch_size', type=int, default=8,
                        help='Number of tasks in a mini-batch of tasks.')
    parser.add_argument('--num_meta_train_tasks', type=int, default=1000,
                        help='Number of meta tasks the model is trained over.')
    parser.add_argument('--num_meta_test_tasks', type=int, default=300,
                        help='Number of meta tasks the model is tested over.')

    args_MAML = parser.parse_args()
    args_MAML.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # """Returns a args configuration for MAML."""
    # args_MAML = {'update_lr': 0.1,     # learner中的学习率，即\alpha
    #              'meta_lr': 0.003,     # meta-learner的学习率，即\beta
    #              'n_way': 5,           # C-way
    #              'k_shot': 5,          # K-shot
    #              'k_query': 15,        # M-test
    #              'task_num': 4,        # 每轮抽4个任务进行训练
    #              'update_step': 5,     # task-level inner update steps
    #              'update_step_test': 5 # 用于finetunning
    #              }
    return args_MAML

def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())

def weights_init_2(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def weights_init_1(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.2)
        nn.init.constant_(m.bias, 0.25)
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data, 0.0, 0.2)
        nn.init.constant_(m.bias, 0.25)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0.0)

def init_data_folders(dataset='UPB', class_num=7, num_samples_per_class=300):
    # init character folders for dataset construction
    if dataset == 'UPB':
        datapath = '/home/chen/workspace/peixinglong/code_for_pxl1/tempdata/UPB/7way.pkl'
        # datapath = '.\\tempdata\\' + 'UPB\\' + str(class_num) + 'way' + '.pkl'#!!!!!!!
        logging.info("datset: " + dataset + '   datapath: ' + datapath)
        if not os.path.exists("%s" % datapath):
            train_character_folders, test_character_folders = tg.UPBFolders(class_num)
            os.makedirs('.\\tempdata\\UPB')
            with open(datapath, 'wb') as handle:
                pickle.dump((train_character_folders, test_character_folders), handle)
        else:
            logging.info("The data folders of the datapath already exists. Load directly.")
            with open(datapath, 'rb') as pkl:
                train_character_folders, test_character_folders = pickle.load(pkl)
        return train_character_folders, test_character_folders

    elif dataset == 'CWRU':
        datapath = '.\\tempdata\\' + 'CWRU\\' + str(10) + 'way' + '.pkl'
        logging.info("datset: " + dataset + '   datapath: ' + datapath)
        if not os.path.exists("%s" % datapath):
            character_folders = tg.CWRUFolders(num_samples_per_class=num_samples_per_class)
            os.makedirs('.\\tempdata\\CWRU')
            with open(datapath, 'wb') as handle:
                pickle.dump((character_folders), handle)
        else:
            logging.info("The data folders of the datapath already exists. Load directly.")
            with open(datapath, 'rb') as pkl:
                character_folders = pickle.load(pkl)
        return character_folders

    elif dataset == 'SEU':
        datapath = '.\\tempdata\\' + 'SEU\\' + str(9) + 'way' + '.pkl'
        logging.info("datset: " + dataset + '   datapath: ' + datapath)
        if not os.path.exists("%s" % datapath):
            character_folders = tg.SEUFolders(num_samples_per_class=num_samples_per_class)
            os.makedirs('.\\tempdata\\SEU')
            with open(datapath, 'wb') as handle:
                pickle.dump((character_folders), handle)
        else:
            logging.info("The data folders of the datapath already exists. Load directly.")
            with open(datapath, 'rb') as pkl:
                character_folders = pickle.load(pkl)
        return character_folders

    else:
        pass


