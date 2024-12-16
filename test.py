#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
# from ns3gym import ns3env

import re
import os
import pickle
import json
import logging

import torch
import numpy as np
import gym
from gym import spaces
# import gymnasium
# from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from py_utlis.general_function import setlogger, setup_seed
from py_utlis.models import CustomNs3Env
from py_utlis.models import CustomActorCriticPolicy


# __author__ = "Piotr Gawlowicz"
# __copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
# __version__ = "0.1.0"
# __email__ = "gawlowicz@tkn.tu-berlin.de"


# startSim = True# False
# iterationNum = 1 #1 # ！！需要修改 espoch次数
# port = 5555
# simTime = 100000 # seconds
# stepTime = 0.5  # seconds
# seed = 0
# simArgs = {"--simTime": simTime,
#            "--stepTime": stepTime,
#            "--testArg": 123}
# debug = False

# env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
# env.reset()

# ob_space = env.observation_space
# ac_space = env.action_space
# print("Observation space: ", ob_space,  ob_space.dtype)
# print("Action space: ", ac_space, ac_space.dtype)


def get_env():
    setup_seed(seed=1)

    startSim = True# False
    iterationNum = 1 #1 # ！！需要修改 espoch次数
    port = 5555
    simTime = 10000000 # seconds
    stepTime = 0.1  # seconds
    seed = 0
    simArgs = {"--simTime": simTime,
            "--stepTime": stepTime,
            "--testArg": 123}
    debug = False

    env = CustomNs3Env(path_reward_save, port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
    # env.reset()

    ob_space = env.observation_space
    # ac_space = env.action_space
    m_n_max_VNF = int(ob_space['box_VNR_2'].shape[0] / 2)
    m_n_nodes = int(ob_space['box_net_2'].shape[0] / 4)

    ac_space_re = spaces.MultiDiscrete(m_n_nodes * np.ones(m_n_max_VNF).astype(int))
    # ac_space_re = spaces.MultiDiscrete(np.full((m_n_max_VNF,), m_n_nodes))

    env.action_space = ac_space_re

    print("Observation space: ", ob_space, ob_space.dtype)
    print("Action space: ", ac_space_re, ac_space_re.dtype)

    env.total_steps = 512 * 250 #512*1500  # 1500 episodes => 768000## 2000 episodes => 1024000

    # check_env(env, warn=True)# 检查env
    # env.reset()
    # action = env.action_space.sample()
    # next_state1, reward1, done1, _, = env.step(action)
    return env

def model_train(env):

    logging.info("Begin medel_train")

    # net_arch = [128, 256, dict(pi=[256], vf=[64])]#自制的神经网络架构来替代A2C，PPO这些算法里面原来的神经网络架构
    model = PPO(
        policy=CustomActorCriticPolicy, #policy='MlpPolicy', #CustomActorCriticPolicy,  #MlpPolicy, 定义策略网络为MLP网络                       # MlpPolicy定义策略网络为MLP网络
        env=env,
        learning_rate=3e-4,
        n_steps=512,#168=672/4 #504 #512
        batch_size=64,#63 #64
        n_epochs=20,
        gamma=0.98,
        gae_lambda=0.95,
        # policy_kwargs={"net_arch": net_arch},
        verbose=1,                                   # verbose=1代表打印训练信息，如果是0为不打印，2为打印调试信息
        tensorboard_log=path_model_train_log,  # 训练数据保存目录，可以用tensorboard查看
        seed=None
    )
    model.learn(total_timesteps=env.total_steps)

    model.save(model_save_pkl)
    # del model  # remove to demonstrate saving and loading
    # with open(env_save_pkl, 'wb') as handle:
    #     json.dump((env), handle)
    with open(info_save_pkl, 'wb') as handle:
        pickle.dump((env.info_list), handle)

    return model


def model_eval():

    # # obs = env.state
    # obs = env.reset()
    #
    # num_dry_run = 1 #空跑1次
    # for ndr in range(num_dry_run):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)

    with open(info_save_pkl, 'rb') as pkl:
        info_list = pickle.load(pkl)

    dict_num_try_result = dict()
    num_try = 1#尝试10个episode
    for k_episode in range(num_try):

        # eval
        env.e_cnt_step = []
        env.acc_list = []
        env.Rev_N_list = []
        env.Cost_N_list = []
        env.Rev_to_Cost_N_list = []

        for j in range(512):
            # action, _states = model.predict(obs)
            # # obs, rewards, dones, info = env.eval_env_step(action)
            # obs, rewards, dones, info = env.step(action)
            info = info_list[j]

            ### eval: e_cnt_step, e_acc, e_Cost_N, e_Rev_N
            pattern = r'(\w+)=([\d.]+)'
            matches = re.findall(pattern, info)
            var_dict = {k: float(v) for k, v in matches}
            e_cnt_step = var_dict['e_cnt_step']
            e_acc = var_dict['e_acc']
            e_Rev_N = var_dict['e_Rev_N']
            e_Cost_N = var_dict['e_Cost_N']

            env.e_cnt_step.append(e_cnt_step)
            env.acc_list.append(e_acc)
            env.Rev_N_list.append(e_Rev_N)
            env.Cost_N_list.append(e_Cost_N)
            # env.Rev_to_Cost_N_list.append(e_Rev_N / e_Cost_N)

            print(str(j))

        acc = np.array(env.acc_list)
        Rev_N = np.array(env.Rev_N_list)
        Cost_N = np.array(env.Cost_N_list)
        # Rev_to_Cost_N = np.array(env.Rev_to_Cost_N_list)

        acc_rate = acc.mean()
        L_A_Rev_N = Rev_N.mean()
        L_A_Cost_N = Cost_N.mean()
        L_A_revenue_cost_ratio = Rev_N.sum() / Cost_N.sum()

        print('acc_rate: '+str(acc_rate)+'; '
              + 'L_A_Rev_N: ' + str(L_A_Rev_N) + '; '
              + 'L_A_Cost_N: ' + str(L_A_Cost_N) + '; '
              + 'L_A_revenue_cost_ratio: ' + str(L_A_revenue_cost_ratio) + '; '
              )

        # dict_num_try_result[k_episode]={'average_switch_F_trans_time': average_switch_F_trans_time,
        #                                 'average_task_completion_time': average_task_completion_time}

        print('k_episode= '+str(k_episode))

    # list_dr = []
    # for i in range(len(dict_num_try_result)):
    #     dr = dict_num_try_result[i]
    #     print('k_episode='+str(i)+' --> '+ 'average_task_completion_time= '  +str(dr['average_task_completion_time']))
    #     list_dr.append(dr['average_task_completion_time'])
    # print('mean = '+str(np.array(list_dr).mean())+'; '+'var = '+str(np.array(list_dr).var()))

    print('Eval End')


if __name__ == "__main__":

    # ---------------------------path set------------------------
    root = os.path.dirname(os.path.abspath(__file__))
    result_root = root + '/result'

    path_model_train_log = os.path.join(result_root, 'model_train_log')
    if not os.path.exists("%s" % path_model_train_log):
        os.makedirs("%s" % path_model_train_log)

    path_model_save = os.path.join(result_root, 'model_save')
    if not os.path.exists("%s" % path_model_save):
        os.makedirs("%s" % path_model_save)

    path_env_save = os.path.join(result_root, 'env_save')
    if not os.path.exists("%s" % path_env_save):
        os.makedirs("%s" % path_env_save)

    path_reward_save = os.path.join(result_root, 'reward_save')
    if not os.path.exists("%s" % path_reward_save):
        os.makedirs("%s" % path_reward_save)

    path_info_save = os.path.join(result_root, 'info_save')
    if not os.path.exists("%s" % path_info_save):
        os.makedirs("%s" % path_info_save)

    path_logger = os.path.join(result_root, 'logger_save')
    if not os.path.exists("%s" % path_logger):
        os.makedirs("%s" % path_logger)

    # ----------------------------logging------------------------
    setlogger(os.path.join(path_logger, 'train.log'))
    f = open("%s/train.log" % path_logger, 'w')
    f.truncate()
    f.close()

    logging.info("Create the system save path of model and env")

    attach_word = 'GP_lc_'#'_LinkNum_0.15_FlowNum_0.15_'#'_'# ！！!
    model_save_pkl = os.path.join(path_model_save, attach_word + 'model_save.pkl')
    env_save_pkl = os.path.join(path_env_save, attach_word + 'env_save.pkl')
    info_save_pkl = os.path.join(path_info_save, attach_word + 'info_save.pkl')

    env = get_env()

    if not os.path.exists("%s" % model_save_pkl) or not os.path.exists("%s" % info_save_pkl):
        logging.info("———————生成env和DRL_model—————")
        model = model_train(env)
    else:
        logging.info("The data folders of the datapath already exists. Load directly.")
        model = PPO.load(model_save_pkl)
        # with open(env_save_pkl, 'rb') as pkl:
        #     env = json.load(pkl)

    model_eval()





# stepIdx = 0
# currIt = 0

# try:
#     while True:
#         print("Start iteration: ", currIt)
#         obs = env.reset()
#         print("Step: ", stepIdx)
#         print("---obs: ", obs)

#         while True:
#             print("--->**** test.py: while once begin, stepIdx = ", stepIdx)
#             stepIdx += 1
#             action = env.action_space.sample()
#             print("---action: ", action)

#             print("Step: ", stepIdx)
#             obs, reward, done, info = env.step(action)

#             print("--->**** env.step() over")

#             print("---obs, reward, done, info: ", obs, reward, done, info)
#             # key_box_VNR_1 = obs["key_box_VNR_1"]
#             # key_box_net_1 = obs["key_box_net_1"]
#             # print("---key_box_VNR_1: ", key_box_net_1)
#             # print("---key_box_net_1: ", key_box_net_1)

#             if done:
#                 stepIdx = 0
#                 if currIt + 1 < iterationNum:
#                     env.reset()
#                 break

#             # print("--->**** test.py: while once end, stepIdx = ", stepIdx)

#         currIt += 1
#         if currIt == iterationNum:
#             break


# except KeyboardInterrupt:
#     print("Ctrl-C -> Exit")
# finally:
#     env.close()
#     print("Done")
