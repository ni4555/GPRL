/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Technische Universität Berlin
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz@tkn.tu-berlin.de>
 */


#ifndef MY_GYM_ENTITY_H
#define MY_GYM_ENTITY_H

#include "utlis.h"
#include "ns3/opengym-module.h"
#include "ns3/nstime.h"



namespace ns3 {

class MyGymEnv : public OpenGymEnv
{
public:
  MyGymEnv ();

  MyGymEnv (NodeContainer& nodes,
            std::map<std::string, NetDeviceContainer>& device_container_map,
            std::map<std::string, Ipv4InterfaceContainer>& ip_interface_container_map,
            std::vector<NodeAttr>& node_attr_vector,
            std::vector<std::vector<uint32_t>> node_VNF_deployment_vector,
            std::map<std::string, LinkAttr>& link_attr_map,
            std::map<std::string, uint32_t> link_str_to_idx_map,
            std::vector<GraphVNR>& graphVNR_vector,
            // Ptr<FlowMonitor>& monitor,
            // Ptr<Ipv4FlowClassifier>& classifier,
            double simulationTime,
            uint32_t stepTotalNumber);

  virtual ~MyGymEnv ();
  static TypeId GetTypeId (void);
  virtual void DoDispose ();

  Ptr<OpenGymSpace> GetActionSpace();
  Ptr<OpenGymSpace> GetObservationSpace();
  bool GetGameOver();
  Ptr<OpenGymDataContainer> GetObservation();
  float GetReward();
  std::string GetExtraInfo();
  bool ExecuteActions(Ptr<OpenGymDataContainer> action);

  void TraceSinkNetStats(std::string context, Ptr<const Packet> macTxTrace);
  void GetNetStats();
  void ResetReadNetStats();
  void ExeNetApp(uint32_t id_from_node_net, uint32_t id_to_node_net, double bandw_embed);

  // Adding attributes
  uint32_t m_n_nodes;
  uint32_t m_n_links;
  uint32_t m_n_type_VNF;
  uint32_t m_n_max_VNF;
  uint32_t m_n_min_VNF;
  uint32_t m_n_max_links_VNR;
  uint32_t m_n_nodes_VNR;
  uint32_t m_n_links_VNR;
  std::vector<NodeAttr> m_node_attr_vector;
  std::vector<std::vector<uint32_t>> m_node_VNF_d_vector;
  std::map<std::string, LinkAttr> m_link_attr_map;
  std::map<std::string, uint32_t> m_link_str_to_idx_map;
  std::vector<std::string> m_link_idx_to_str_vector;
  std::vector<uint32_t> m_from_node_vector;
  std::vector<uint32_t> m_to_node_vector;
  std::map<uint32_t, std::vector<std::string> > m_nodeId_to_DevStr_map;
  uint32_t m_port_onoff;

  std::vector<GraphVNR> m_graphVNR_vector;
  GraphVNR m_graphVNR;
  double m_life_cycle;
  std::vector<double> m_interval_vector;
  std::vector<double> m_life_cycle_vector;

  // 网络状态
  uint32_t m_setPacketSize;
  bool m_log_Tx_read;
  double m_tau_Tx_read;
  std::map<uint32_t, std::map<uint32_t, NodeStepCache> > m_node_cache_map;
  // std::vector<std::vector<NodeStepCache> > m_node_cache_vector;
  std::map<std::string, double> m_link_stats_map;
  std::vector<double> m_link_dateRate_vector; // 向量索引为link id, value为Tx网口速率。由于不断更新数值，存储的数据为截至当前最后一次测得的网口速率
  // std::map<uint32_t, std::vector<std::string>> m_link_TxTime_map; // key为link id, value中的仅有两个元素{m_prevTxTime, m_curTxTime}
  std::vector<double> m_link_curTxTime_vector; // 向量索引为link id, value为完成当前发包的时间。
  // std::vector<double> m_link_prevTxTime_vector; // 向量索引为link id, value为完成上一个发包的时间。
  // std::map<std::string, double> m_Tx_tau_map; // 注意在变量命名时，tau与Time均为时刻。key为link str，value为发包完成的时刻
  std::map<std::string, uint32_t> m_Tx_packetSizeSum_map;// key为link str，value为发包大小的累积
  std::map<std::string, uint32_t> m_Tx_cnt_TxPacket_map; // 记录从Tx_read开始后的发包次数


  // Refactor
  NodeContainer m_nodes;
  std::map<std::string, NetDeviceContainer> m_devs_cont_map;
  std::map<std::string, Ipv4InterfaceContainer> m_ip_interface_cont_map;

  // Ptr<Ipv4FlowClassifier> m_classifier;
  // Ptr<FlowMonitor> m_monitor;

  double m_simulationTime;
  uint32_t m_stepTotalNumber;
  uint32_t m_state_buffer_size;

  // state_fixed
  bool m_log_state_fixed;
  std::map<uint32_t, std::vector<NodeAttr>> m_node_attr_vector_sfmap;
  std::map<uint32_t, std::map<uint32_t, std::map<uint32_t, NodeStepCache>>> m_node_cache_map_sfmap;
  std::map<uint32_t, double> m_l_rand_std_R1_map;
  std::map<uint32_t, double> m_l_rand_std_R2_map;

  // reward
  GraphVNR m_rwd_graphVNR;
  uint32_t m_rwd_cnt_VNR;
  double m_rwd_interval;
  double m_rwd_life_cycle;
  std::deque<double> m_acc_qeque;
  std::deque <double> m_Re_qeque;
  std::deque <double> m_Cost_qeque;

  std::vector <double> m_avg_node_util_R1_vector;
  std::vector <double> m_avg_node_util_R2_vector;

  // action
  std::vector<uint32_t> m_action_vector;

  // eval
  uint32_t e_cnt_step;
  double e_acc;//1为接受；0为放弃
  double e_Rev_N;
  double e_Cost_N;

  // log
  bool m_log_typeVNF;
  bool m_log_nodeRes;

  uint32_t m_cnt_VNR;
  uint32_t m_cnt_step;
  int m_cnt_step_int;
  // int total_number_VNR;

  double m_curTime;
  double m_prevTime;
  double m_curTxTime;
  double m_prevTxTime;
  double m_monitor_interval_time;
  uint32_t m_prev_rxBytes;
  uint32_t m_prev_txBytes;

private:
  void ScheduleNextStateRead();

  double m_interval;
};

}


#endif // MY_GYM_ENTITY_H
