#!/usr/bin/python
# -*- coding:utf-8 -*-
import re
import math
import logging
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import dgl
import dgl.nn.pytorch.conv as dglnn
import networkx as nx
import gym
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.preprocessing import get_action_dim#, is_image_space, maybe_transpose, preprocess_obs
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from ns3gym.ns3env import Ns3Env, Ns3ZmqBridge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomNs3Env(Ns3Env):

    def __init__(self, path_reward_save, stepTime=0, port=0, startSim=True, simSeed=0, simArgs={}, debug=False):
        super().__init__(stepTime=stepTime, port=port, startSim=startSim, simSeed=simSeed, simArgs=simArgs, debug=debug)
        logging.info("Create and initialize the env")
        self.path_reward_save = path_reward_save
        self.total_steps = None
        self.step_cnt = 0
        self.reward_list = []
        self.info_list = []
        # self.state_list = []  #!#
        self.done_list = []
        self.state_fixed_buffer = {}
        self.state_buffer_size = 1000 # ！！应于mygym.cc里面的值相同
        self.log_state_fixed = False#True # ！！应于mygym.cc里面的值相同

    def step(self, action):
        # logging.info("Step the env once: 'step_cnt' is %d" % self.step_cnt)
        print("Step the env once: 'step_cnt' is %d" % self.step_cnt)

        response = self.ns3ZmqBridge.step(action)
        self.envDirty = True

        state, reward, done, info = self.get_state()

        # state_fixed
        if self.log_state_fixed == True:
            if (len(self.state_fixed_buffer) <= self.state_buffer_size):
                self.state_fixed_buffer[self.step_cnt] = state
            else:
                state = self.state_fixed_buffer[self.step_cnt % self.state_buffer_size]

        # eval
        if self.step_cnt + 1 > self.total_steps - 512 * 1 and self.step_cnt + 1 <= self.total_steps:
            self.info_list.append(info)
        #

        info = {'testInfo': info, 'TimeLimit.truncated': False}
        print("reward = %.4f" % reward)

        self.reward_list.append(reward)
        self.done_list.append(done)
        if self.step_cnt + 1 == self.total_steps:
            with open(self.path_reward_save + '/' + str(self.step_cnt + 1) + '_reward_GP_lc_20.pkl',# ！！!
                      'wb') as handle:
                pickle.dump((self.reward_list, self.done_list), handle)

        if ((self.step_cnt + 1) % (512 * 50)) == 0:
            with open(self.path_reward_save + '/reward_GP.pkl',
                      'wb') as handle:
                pickle.dump((self.reward_list, self.done_list), handle)
        # #!#
        # self.state_list.append(state)
        # #!#

        self.step_cnt += 1
        return state, reward, done, info

    def reset(self):
        # logging.info("************ Reset the env once ************")
        print("Step the env once: 'step_cnt' is %d" % self.step_cnt)

        if not self.envDirty:
            obs = self.ns3ZmqBridge.get_obs()
            return obs

        if self.ns3ZmqBridge:
            self.ns3ZmqBridge.close()
            self.ns3ZmqBridge = None

        self.envDirty = False
        self.ns3ZmqBridge = Ns3ZmqBridge(self.port, self.startSim, self.simSeed, self.simArgs, self.debug)
        self.ns3ZmqBridge.initialize_env(self.stepTime)
        self.action_space = self.ns3ZmqBridge.get_action_space()
        self.observation_space = self.ns3ZmqBridge.get_observation_space()
        # get first observations
        self.ns3ZmqBridge.rx_env_state()
        obs = self.ns3ZmqBridge.get_obs()

        #
        self.reward_list = []
        self.info_list = []
        self.done_list = []
        self.step_cnt = 0
        return obs

    # def eval_env_step(self, action):
    #     # logging.info("Step the env once: 'step_cnt' is %d" % self.step_cnt)
    #     print("Step the env once: 'step_cnt' is %d" % self.step_cnt)
    #
    #     response = self.ns3ZmqBridge.step(action)
    #     self.envDirty = True
    #
    #     state, reward, done, info = self.get_state()
    #
    #     # state_fixed
    #     if self.log_state_fixed == True:
    #         if (len(self.state_fixed_buffer) <= self.state_buffer_size):
    #             self.state_fixed_buffer[self.step_cnt] = state
    #         else:
    #             state = self.state_fixed_buffer[self.step_cnt % self.state_buffer_size]
    #
    #     info = {'testInfo': info, 'TimeLimit.truncated': False}
    #     print("reward = %.4f" % reward)
    #
    #     self.reward_list.append(reward)
    #     self.done_list.append(done)
    #     if self.step_cnt + 1 == self.total_steps:
    #         with open(self.path_reward_save + '/' + str(self.step_cnt) + '_reward_XX.pkl',
    #                   'wb') as handle:
    #             pickle.dump((self.reward_list, self.done_list), handle)
    #
    #     if (self.step_cnt % (512 * 50)) == 0:
    #         with open(self.path_reward_save + '/reward_XX.pkl',
    #                   'wb') as handle:
    #             pickle.dump((self.reward_list, self.done_list), handle)
    #
    #     ### eval: e_cnt_step, e_acc, e_Cost_N, e_Rev_N
    #     pattern = r'(\w+)=([\d.]+)'
    #     matches = re.findall(pattern, info)
    #     var_dict = {k: float(v) for k, v in matches}
    #     e_cnt_step = var_dict['e_cnt_step']
    #     e_acc = var_dict['e_acc']
    #     e_Rev_N = var_dict['e_Rev_N']
    #     e_Cost_N = var_dict['e_Cost_N']
    #
    #     self.e_cnt_step.append(e_cnt_step)
    #     self.acc_list.append(e_acc)
    #     self.Rev_N_list.append(e_Rev_N)
    #     self.Cost_N_list.append(e_Cost_N)
    #     self.Rev_to_Cost_N_list.append(e_Rev_N/e_Cost_N)
    #
    #     self.state = state
    #     ###
    #
    #     self.step_cnt += 1
    #     return state, reward, done, info


class ExtractFeaturesNetwork(nn.Module):

    def __init__(self):
        super(ExtractFeaturesNetwork, self).__init__()
        self.layer1 = nn.Sequential()
        # self.g_VNR = dgl.DGLGraph()
        # self.g_net = dgl.DGLGraph()

    def forward(self, x):
        x = self.layer1(x)
        # 处理state数据 => DGLGraph图数据  batch处理图
        b_size = x['box_VNR_1'].size()[0] # 一个batch的图数量
        set_b_n_VNR = torch.zeros(b_size)
        set_b_e_VNR = torch.zeros(b_size)
        set_b_n_net = torch.zeros(b_size)
        set_b_e_net = torch.zeros(b_size)

        for i_b in range(b_size):
            # VNR
            n_n_VNR = x['box_VNR_1'][i_b, 0].to(torch.int64).item()
            n_e_VNR = x['box_VNR_1'][i_b, 2].to(torch.int64).item()
            n_nfeat_VNR = x['box_VNR_1'][i_b, 1].to(torch.int64).item()
            n_efeat_VNR = x['box_VNR_1'][i_b, 3].to(torch.int64).item()
            assert max(x['box_VNR_1'][i_b, 4:]) + 1 == n_n_VNR, "VNR的节点数目异常"

            u_VNR = x['box_VNR_1'][i_b, 4:4 + n_e_VNR].to(torch.int64)
            v_VNR = x['box_VNR_1'][i_b, 4 + n_e_VNR:4 + n_e_VNR * 2].to(torch.int64)
            g_VNR = dgl.graph((u_VNR, v_VNR))
            # plt.figure()
            # nx.draw_networkx(g_VNR.to_networkx(), arrows=None, with_labels=True)
            # plt.show()
            g_VNR.ndata['nfeat'] = x['box_VNR_2'][i_b, :n_n_VNR * n_nfeat_VNR].reshape(n_n_VNR, n_nfeat_VNR)
            g_VNR.ndata['nmask'] = torch.ones(n_n_VNR).unsqueeze(1)
            g_VNR.edata['efeat'] = x['box_VNR_3'][i_b, :n_e_VNR * n_efeat_VNR].reshape(n_e_VNR, n_efeat_VNR)
            g_VNR.ndata['vnr_VNF'] = x['box_VNR_VNF'][i_b, :n_n_VNR].reshape(n_n_VNR, 1)

            n_n_mask = int(x['box_VNR_2'].size()[-1] / n_nfeat_VNR - n_n_VNR)
            g_VNR.add_nodes(n_n_mask)#, {'nfeat': torch.zeros(n_n_mask, n_nfeat_VNR), 'nmask': torch.zeros(n_n_mask)})

            # net
            n_n_net = x['box_net_1'][i_b, 0].to(torch.int64).item()
            n_e_net = x['box_net_1'][i_b, 2].to(torch.int64).item()
            n_nfeat_net = x['box_net_1'][i_b, 1].to(torch.int64).item()
            n_efeat_net = x['box_net_1'][i_b, 3].to(torch.int64).item()
            assert max(x['box_net_1'][i_b, 4:])+1 == n_n_net, "net的节点数目异常"

            # mid_index = x['box_net_1'][0, 4:].size(0) // 2
            u_net = x['box_net_1'][i_b, 4:4 + n_e_net].to(torch.int64)
            v_net = x['box_net_1'][i_b, 4 + n_e_net:4 + n_e_net * 2].to(torch.int64)
            g_net = dgl.graph((u_net, v_net))
            # plt.figure()
            # nx.draw_networkx(g_net.to_networkx(), arrows=None, with_labels=True)
            # plt.show()
            g_net.ndata['nfeat'] = x['box_net_2'][i_b, :n_n_net * n_nfeat_net].reshape(n_n_net, n_nfeat_net)
            g_net.edata['efeat'] = x['box_net_3'][i_b, :n_e_net * n_efeat_net].reshape(n_e_net, n_efeat_net)
            g_net.ndata['net_VNF'] = x['box_net_VNF'][i_b, :].reshape(n_n_net, -1)

            # batch
            set_b_n_VNR[i_b] = n_n_VNR + n_n_mask # ！！
            set_b_e_VNR[i_b] = n_e_VNR
            set_b_n_net[i_b] = n_n_net
            set_b_e_net[i_b] = n_e_net

            if i_b > 0:
                dg_VNR = dgl.batch([dg_VNR, g_VNR])
                dg_net = dgl.batch([dg_net, g_net])
            else:
                dg_VNR = dgl.batch([g_VNR])
                dg_net = dgl.batch([g_net])

        dg_VNR.set_batch_num_nodes(set_b_n_VNR.to(torch.int64))
        dg_VNR.set_batch_num_edges(set_b_e_VNR.to(torch.int64))
        dg_net.set_batch_num_nodes(set_b_n_net.to(torch.int64))
        dg_net.set_batch_num_edges(set_b_e_net.to(torch.int64))

        # self.g_VNR = None
        # self.g_net = None
        return (dg_VNR, dg_net)

class EncoderNet(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(EncoderNet, self).__init__()
        # self.gconv = dglnn.GraphConv(input_size, hidden_size, norm='both', weight=True, bias=True, activation=F.relu, allow_zero_in_degree=True)
        # self.gconv = dglnn.GraphConv(input_size, hidden_size, norm='both', weight=True, bias=True, activation=None,
        #                              allow_zero_in_degree=True)

        self.gconv_1 = dglnn.GATConv(input_size, hidden_size, num_heads=1, bias=True, activation=None,
                                       allow_zero_in_degree=True)
        self.gconv_2 = dglnn.GATConv(hidden_size, hidden_size, num_heads=1, bias=True, activation=None,
                                       allow_zero_in_degree=True)
        self.gconv_3 = dglnn.GATConv(hidden_size, hidden_size, num_heads=1, bias=True, activation=None,
                                       allow_zero_in_degree=True)


    def forward(self, g, feat, efeat):
        # out = self.gconv(graph=g, feat=feat, edge_weight=efeat)
        out = self.gconv_1(graph=g, feat=feat, edge_weight=None)
        out = self.gconv_2(graph=g, feat=out, edge_weight=None)
        out = self.gconv_3(graph=g, feat=out, edge_weight=None)

        return out  # (batch, hidden_size, seq_len)

class AttentionModule(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(AttentionModule, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, static_hidden, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        # attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns


class PointerNet(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.0):
        super(PointerNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=False)#,
                          # dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = AttentionModule(hidden_size)

        # self.drop_rnn = nn.Dropout(p=dropout)
        # self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, vx_hidden, last_hh):

        rnn_out, last_hh = self.gru(vx_hidden.permute(2, 0, 1), last_hh) #进行RNN
        # rnn_out = vx_hidden.permute(2, 0, 1) # 不进行RNN
        # nn.GRU的第一个输入变量的维度：（seq_len,batch,input_size），第二个输入变量的维度：（D*num_layer,batch,hidden_size）

        # # Always apply dropout on the RNN output
        # rnn_out = self.drop_rnn(rnn_out)
        # if self.num_layers > 1:
        #     # If > 1 layer dropout is already applied
        #     last_hh = self.drop_hh(last_hh)

        rnn_out = rnn_out.squeeze(0)

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, rnn_out)

        # ####
        # context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)
        #
        # # Calculate the next output using Batch-matrix-multiply ops
        # context = context.transpose(1, 2).expand_as(static_hidden)
        # energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)
        #
        # v = self.v.expand(static_hidden.size(0), -1, -1)
        # W = self.W.expand(static_hidden.size(0), -1, -1)
        #
        # # probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)
        # probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy)))
        #
        # return probs, last_hh
        #####

        return enc_attn, last_hh


class GraphEncoderNet(nn.Module):
    """
    static为network
    vnr为VNR
    """

    def __init__(self, static_size, vnr_size, hidden_size):
        super(GraphEncoderNet, self).__init__()
        self.static_size = static_size
        self.vnr_size = vnr_size
        self.hidden_size = hidden_size

        self.static_encoder = EncoderNet(static_size, hidden_size)
        self.vnr_encoder = EncoderNet(vnr_size, hidden_size)

        # for p in self.parameters():
        #     if len(p.shape) > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, graphs_tuple):
        """
        ---维度---
        static_feat: (batch_size, num_feats_net, num_net_nodes)
        vnr_feat: (batch_size, num_feats_vnr, num_vnr_nodes)
        vx_hidden: (batch_size, num_feats_vnr, 1)
        """
        g_vnr, g_static = graphs_tuple
        # static_feat = g_static.ndata['nfeat'].unsqueeze(0).transpose(1, 2)
        # vnr_feat = g_vnr.ndata['nfeat'].unsqueeze(0).transpose(1, 2)
        static_hidden = self.static_encoder(g_static, g_static.ndata['nfeat'], g_static.edata['efeat']).reshape(g_static.batch_size, -1, self.hidden_size).transpose(1, 2)
        vnr_hidden = self.vnr_encoder(g_vnr, g_vnr.ndata['nfeat'], g_vnr.edata['efeat']).reshape(g_vnr.batch_size, -1, self.hidden_size).transpose(1, 2)

        return vnr_hidden, static_hidden


class GraphPinterNet(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """
    """
    static为network
    vnr为VNR
    """
    def __init__(self, hidden_size, num_layers=1, dropout=0):
        super(GraphPinterNet, self).__init__()

        # self.update_fn = update_fn
        # self.mask_fn = None

        # Define the encoder
        # self.static_encoder = EncoderNet(static_size, hidden_size)
        # self.vnr_encoder = EncoderNet(vnr_size, hidden_size)
        # self.gcn_encoder = GraphEncoderNet(static_size, vnr_size, hidden_size)#不共享时启用

        self.pointer = PointerNet(hidden_size, num_layers, dropout)

        # for p in self.parameters():
        #     if len(p.shape) > 1:
        #         nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, hidden_size, 1), requires_grad=True, device=device)

    def forward(self, vnr_hidden, static_hidden, graphs_tuple):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        """
        ---维度---
        static_hidden: (batch_size, num_hidden_net, num_net_nodes)
        vnr_hidden: (batch_size, num_hidden_vnr, num_vnr_nodes)
        vx_hidden: (batch_size, num_feats_vnr, 1)
        """
        batch_size, hidden_size, sequence_size = static_hidden.size()

        last_hh = None
        if last_hh is None:
            last_hh = self.x0.expand(batch_size, -1, -1).permute(2, 0, 1)

        g_vnr, g_static = graphs_tuple

        # vnr_VNF = g_vnr.ndata['vnr_VNF'].to(torch.int64)
        # net_VNF = g_static.ndata['net_VNF'].to(torch.int64)
        #
        # max_steps = vnr_hidden.size()[-1]
        # probs_mask = torch.zeros(max_steps, sequence_size, device=device)

        n_vnf_type = g_static.ndata['net_VNF'].size()[-1]
        vnr_VNF = g_vnr.ndata['vnr_VNF'].squeeze(1).reshape(batch_size, -1).to(torch.int64)
        net_VNF = g_static.ndata['net_VNF'].unsqueeze(0).reshape(batch_size, -1, n_vnf_type).to(torch.int64)

        max_steps = vnr_hidden.size()[-1]
        probs_mask = torch.zeros(batch_size, max_steps, sequence_size, device=device)

        for i in range(max_steps):
            # if not mask.byte().any():
            #     break

            # vnf_r = vnr_VNF[i][0].item()
            # mask = net_VNF[:, vnf_r:vnf_r+1].transpose(1, 0)
            vr = net_VNF[:, :, vnr_VNF[:, i]]
            mask = torch.stack([vr[iv][:, iv] for iv in range(batch_size)], dim=0)

            vx_hidden = vnr_hidden[:, :, i:i + 1]
            probs, last_hh = self.pointer(static_hidden, vx_hidden, last_hh)

            # probs = F.softmax(probs + mask.log(), dim=1)
            probs_mask[:, i:i + 1, :] = probs + mask.unsqueeze(1).log()

        return probs_mask.view(batch_size, -1) # 多离散空间下self.action_dist.proba_distribution(action_logits=mean_actions)的mean_actions需要碾平操作


class ValueNet(nn.Module):
    """
    static为network
    vnr为VNR
    """
    def __init__(self, hidden_size, out_size, n_net_node, n_max_vnf):
        super(ValueNet, self).__init__()
        self.out_size = out_size

        # Define the encoder
        # self.static_encoder = EncoderNet(static_size, hidden_size)
        # self.vnr_encoder = EncoderNet(vnr_size, hidden_size)
        # self.gconv = dglnn.GraphConv(hidden_size, hidden_size, norm='both', weight=True, bias=True, activation=F.relu, allow_zero_in_degree=True)
        self.gconv = dglnn.GraphConv(hidden_size, hidden_size, norm='both', weight=True, bias=True, activation=None,
                                     allow_zero_in_degree=True)
        # self.gconv = dglnn.GATConv(hidden_size, hidden_size, bias=True, activation=None,
        #                                allow_zero_in_degree=True, num_heads=1)
        self.linear = nn.Linear(hidden_size * 2, out_size)
        # self.linear = nn.Linear(hidden_size * (n_net_node + n_max_vnf), out_size)

        # for p in self.parameters():
        #     if len(p.shape) > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, vnr_hidden, static_hidden, graphs_tuple):
        '''
        :param vnr_hidden:
        :param static_hidden:
        :param graphs_tuple:
                y1 (batch_size, hidden_size)
                y2 (batch_size, hidden_size)
        :return: y (batch_size, out_size) 实际应为（1, out_size）
        '''
        g_vnr, g_static = graphs_tuple

        batch_size, hidden_size, vnr_h_sequence_size = vnr_hidden.size()
        batch_size, hidden_size, static_h_sequence_size = static_hidden.size()

        # y1 = self.gconv(g_vnr, vnr_hidden.transpose(1, 2).reshape(1, batch_size * vnr_h_sequence_size, -1).squeeze(0))
        # y2 = self.gconv(g_static, static_hidden.transpose(1, 2).reshape(1, batch_size * static_h_sequence_size, -1).squeeze(0))
        # y1 = y1.reshape(batch_size, -1, y1.size()[-1])
        # y2 = y2.reshape(batch_size, -1, y2.size()[-1])

        y1 = vnr_hidden.transpose(1, 2) * g_vnr.ndata["nmask"].reshape(batch_size, -1).unsqueeze(2)
        y2 = static_hidden.transpose(1, 2)
        y1 = torch.sum(y1, dim=1) / g_vnr.ndata["nmask"].reshape(batch_size, -1).sum(1).unsqueeze(1)
        y2 = torch.sum(y2, dim=1) / y2.size()[1]
        y = torch.cat((y1, y2), dim=1)
        y = self.linear(y)
        return y

        # y1 = vnr_hidden.transpose(1, 2) * g_vnr.ndata["nmask"].reshape(batch_size, -1).unsqueeze(2)
        # y2 = static_hidden.transpose(1, 2)
        # y = torch.cat((y1.reshape(batch_size, -1), y2.reshape(batch_size, -1)), dim=1)
        # y = self.linear(y)
        # return y


# class ActionResetNet(nn.Module):
#     def __init__(self):
#         super(ActionResetNet, self).__init__()
#
#     def forward(self, x, action_dim):
#         # x =
#
#         return x


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    def __init__(
            self,
            last_layer_dim_pi: int = 128,
            last_layer_dim_vf: int = 64,
            device: str = "cpu"
    ):
        super(CustomNetwork, self).__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # # Shared network
        self.static_size = 4 # ！！net中每个节点的特征数
        self.vnr_size = 2 # VNR中每个节点的特征数
        self.hidden_size = 40 #
        self.n_net_node = 23 #23#！！！23
        self.n_max_vnf = 8 # ！！

        self.shared_net = GraphEncoderNet(self.static_size, self.vnr_size, self.hidden_size)

        # Policy network
        self.policy_net = GraphPinterNet(self.hidden_size, num_layers=1, dropout=0)

        # Value network
        self.value_net = ValueNet(hidden_size=self.hidden_size, out_size=last_layer_dim_vf,
                                  n_net_node=self.n_net_node, n_max_vnf=self.n_max_vnf)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        # return self.policy_net(features), self.value_net(features)

        vnr_hidden, static_hidden = self.shared_net(features)
        return self.policy_net(vnr_hidden, static_hidden, features), self.value_net(vnr_hidden, static_hidden, features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        vnr_hidden, static_hidden = self.shared_net(features)
        return self.policy_net(vnr_hidden, static_hidden, features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        vnr_hidden, static_hidden = self.shared_net(features)
        return self.value_net(vnr_hidden, static_hidden, features)
    # def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
    #     return self.policy_net(features)
    #
    # def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
    #     return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        # observation_space: gym.spaces.Space,
        # action_space: gym.spaces.Space,
        # lr_schedule: Schedule,
        # net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        # activation_fn: Type[nn.Module] = nn.Tanh,
        # *args,
        # **kwargs,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        # # Disable orthogonal initialization
        # self.ortho_init = False

        ##重写父类ActorCriticPolicy的属性与方法
        self.action_net = nn.Sequential(
                                        # nn.Linear(self.mlp_extractor.latent_dim_pi, get_action_dim(self.action_space)),
                                        # # nn.Softmax(dim=-1)
                                        # # nn.Softplus()
                                        # # nn.Sigmoid()
                                        # nn.Tanh()
                                        # # nn.ReLU()
                                        # # nn.LeakyReLU()
                                        # # https://blog.csdn.net/DIPDWC/article/details/112686489
                                        )
        # self.action_net = ActionResetNet()
        self.features_extractor = ExtractFeaturesNetwork()
        self.pi_features_extractor = ExtractFeaturesNetwork()
        self.vf_features_extractor = ExtractFeaturesNetwork()

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork()

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed

        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        latent_pi = self.action_zero_padding(latent_pi)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

        # features = self.features_extractor(obs)##self.extract_features可认为是个输入特征预处理的函数，可自定义重写覆盖它(ExtractFeaturesNetwork())
        # latent_pi, latent_vf = self.mlp_extractor(features)
        # # Evaluate the values for the given observations
        # values = self.value_net(latent_vf)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        # actions = distribution.get_actions(deterministic=deterministic)
        # log_prob = distribution.log_prob(actions)
        # actions = actions.reshape((-1,) + self.action_space.shape)
        # return actions, values, log_prob

    def evaluate_actions(self, obs, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def action_zero_padding(self, latent_pi):
        n_max_vnf = get_action_dim(self.action_space)
        n_n_net = self.action_space.nvec[0]
        n_res = n_max_vnf - int(latent_pi.size()[-1]/n_n_net)
        latent_pi = torch.cat((latent_pi, torch.zeros(1, n_res * n_n_net)), dim=1)
        return latent_pi



###############################################################################

class TCNModel(nn.Module):
    def __init__(self, args):
        super(TCNModel, self).__init__()
        self.transformer_module = TransformerModule(data_length = args.data_length,
                                                    len_patch=args.len_patch,
                                                    ninp=args.ninp,
                                                    nhead=args.nhead,
                                                    nhid=args.nhid,
                                                    nlayers=args.nlayers,
                                                    dropout=args.dropout)
        self.CNN_layer = CNNLayer(data_length=args.data_length,
                                  len_patch=args.len_patch)
        self.classifier_layer = ClassifierLayer(ninp_classifier=64 * 25,
                                                num_classes=args.num_ways)

    def forward(self, x):
        out = self.transformer_module(x)
        return self.classifier_layer(self.CNN_layer(out))


class TransformerModule(nn.Module):
    """Container module with a encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, data_length = 1024, len_patch=32, ninp=200, nhead=4, nhid=200, nlayers=2, dropout=0.2, activation = "relu"):
        super(TransformerModule, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ninp = ninp
        self.len_patch = len_patch
        self.data_length = data_length # BN #！！！！！！未使用
        self.linear_projection = nn.Linear(len_patch, ninp)
        self.BN_layer = nn.BatchNorm1d(self.data_length // self.len_patch)#data_length // len_patch#！！！！！！未使用
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation)
        # encoder_norm = nn.LayerNorm(ninp)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # self.encoder = nn.Embedding(ntoken, ninp)
        # self.decoder = nn.Linear(ninp, ntoken)
        self.Dropout1 = nn.Dropout(dropout)
        # self.Dropout2 = nn.Dropout(dropout)

        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # def init_weights(self):
    #     initrange = 0.1
    #     nn.init.uniform_(self.encoder.weight, -initrange, initrange)
    #     nn.init.zeros_(self.decoder.weight)
    #     nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def src_patch(self,src):
        # src(input) => (N, C, L)
        # output => (C, N, L(len_patch))
        if not isinstance(src, torch.Tensor):
            try:
                src = torch.Tensor(src)
            except:
                raise TypeError("The input of data type for the model should be 'torch.Tensor'.")

        assert src.shape[1] == 1, "The model type (MODELTYPE) for the model should be '1d'."
        assert len(src.shape) == 3,"The dimension of src should be 3."
        # src=>(N, L), where N is the batch size, L is the length of each sample for the model input.
        return src[:, :, :src.shape[-1] // self.len_patch * self.len_patch]\
                    .reshape(src.shape[0],-1, self.len_patch)\
                    .transpose(1,0)

    def data_norm(self, src, norm_type='ZeroMean'):
        if norm_type == 'L2':
            return torch.div(src, torch.norm(src, p=2, dim=-1).unsqueeze(1))
        elif norm_type == 'MinMax':
            return torch.div(src - torch.min(src, dim=-1).values.unsqueeze(1),
                             (torch.max(src, dim=-1).values - torch.min(src, dim=-1).values).unsqueeze(1))
        elif norm_type == 'ZeroMean':
            return torch.div(src - torch.mean(src, dim=-1).unsqueeze(1),
                             torch.std(src, dim=-1).unsqueeze(1))
        else:
            return src

    def forward(self, src, has_mask=True):
        ############ preparation #############
        # Linear Projection of all patches for the signal sample
        # src(input) => (N, C, L), where N is the batch size, C is the number of channels (if 'modeltype' is '1d', C is 1),
        #               L is the length of each sample for the model input (1024).
        # output => (N, S, ninp), where S is the source sequence length.

        ##Input data normalization processing

        #######！！！！！！！！！！！！！！！
        # src = self.data_norm(src,norm_type='ZeroMean')
        ## norm_type: 'L2', 'MinMax', 'ZeroMean'

        if not src.dtype == torch.float32:
            src = src.to(torch.float32)

        src = self.src_patch(src) # (C, N, L(len_patch))
        ## src = self.Dropout1(self.linear_projection(self.src_patch(src)) * math.sqrt(self.ninp))
        src = self.linear_projection(src)
        # src = self.BN_layer(src.transpose(1,0)).transpose(1,0)
        # src = src * math.sqrt(self.ninp)
        ## src = self.encoder(src) * math.sqrt(self.ninp)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.pos_encoder(src)
        # return src.transpose(1, 0)
        # ############ encoder #############
        output = self.transformer_encoder(src, self.src_mask)
        # # ############ decoder #############
        # # output = self.decoder(output)
        # ############ output #############
        return output.transpose(1, 0)


class CNNLayer(nn.Module):

    def __init__(self, data_length, len_patch):
        super(CNNLayer, self).__init__()

        # self.layer1 = nn.Sequential(
        #     nn.Conv1d(data_length // len_patch, 32, kernel_size=5, padding=1, stride=2),
        #     nn.BatchNorm1d(16, momentum=1, affine=True),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2))
        # self.layer2 = nn.Sequential(
        #     nn.Conv1d(16, 16, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm1d(16, momentum=1, affine=True),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2))

        self.layer1 = nn.Sequential(
            nn.Conv1d(data_length // len_patch, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            # nn.MaxPool1d(2)
        )#!!
        # self.layer4 = nn.Sequential(
        #     nn.Conv1d(64, 64, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm1d(64, momentum=1, affine=True),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(2)
        # )#!!

        self.admp = nn.AdaptiveMaxPool1d(25)
        # self.admp = nn.AdaptiveMaxPool1d(8)

    def forward(self, x):
        # input => (N, C, L), where N is the batch size, C is the number of channel, L is the length of features.
        # output => (N, L_out), where L_out is the length after flattened.
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = self.admp(out)
        out = out.view(out.size(0), -1)
        return out


class ClassifierLayer(nn.Module):

    def __init__(self, ninp_classifier=32 * 10, num_classes=5):#128
        super(ClassifierLayer, self).__init__()
        self.ninp_classifier = ninp_classifier
        self.classifier = nn.Linear(ninp_classifier, num_classes)

    def forward(self, x):
        # input => (N, C, L), where N is the batch size, C is the number of channel, L is the length of features.
        # output => (N, L_out), where L_out is equal to num_classes.
        assert self.ninp_classifier == x.shape[-1],\
            "The input data is not equal to the input dimension of ClassifierLayer."
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
