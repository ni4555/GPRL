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

#include "utlis.h"
#include "mygym.h"

#include "ns3/opengym-module.h"
// #include "ns3/nstime.h"

#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/wifi-module.h"
#include "ns3/node-list.h"
#include "ns3/log.h"

#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/csma-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
// #include "ns3/flow-monitor-module.h"

#include "ns3/trace-source-accessor.h"
#include "ns3/traced-value.h"
#include "ns3/uinteger.h"

#include <sstream>
#include <iostream>
#include <cstdint> // 包含 uint32_t 的定义
#include <algorithm>
#include <fstream>
#include <string>
#include <cassert>
#include <cmath>
#include <typeinfo>
#include <vector>
#include <map>
// #include <unordered_map>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MyGymEnv");

NS_OBJECT_ENSURE_REGISTERED (MyGymEnv);

MyGymEnv::MyGymEnv ()
{
  NS_LOG_FUNCTION (this);
  // m_interval = Seconds(0.1);

  // Simulator::Schedule (Seconds(0.0), &MyGymEnv::ScheduleNextStateRead, this);
}

MyGymEnv::MyGymEnv (NodeContainer& nodes,
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
                    uint32_t stepTotalNumber)
// : m_ip_interface_cont_map(ip_interface_container_map)
{
  NS_LOG_FUNCTION (this);
  // ##**##

  m_simulationTime = simulationTime;
  m_stepTotalNumber = stepTotalNumber;
  m_state_buffer_size = 1000;
  m_log_state_fixed = true;

  m_nodes = nodes;
  m_n_nodes = m_nodes.GetN();
  m_n_links =  static_cast<uint32_t>(link_attr_map.size());
  m_n_type_VNF = 5;//！！VNF的种类数目
  m_n_max_VNF = 8;//8！！单个VNR中VNF的最大个数
  m_n_min_VNF = 5;//4！！单个VNR中VNF的最小个数
  m_n_max_links_VNR = static_cast<uint32_t>(static_cast<double>(m_n_max_VNF * (m_n_max_VNF - 1)) / 2); // DAG的最大有向边的数目
  m_devs_cont_map = device_container_map;
  m_ip_interface_cont_map = ip_interface_container_map;
  m_node_attr_vector = node_attr_vector;
  m_node_VNF_d_vector = node_VNF_deployment_vector;
  m_link_attr_map = link_attr_map;
  m_link_str_to_idx_map = link_str_to_idx_map;
  m_link_idx_to_str_vector.resize(m_n_links);

  // 网络状态
  m_setPacketSize = 80; // 50 设定网口发包的大小
  m_link_dateRate_vector.resize(m_n_links);
  m_link_curTxTime_vector.resize(m_n_links);
  // m_link_prevTxTime_vector.resize(m_n_links);
  for (const auto& pair : m_link_str_to_idx_map) {
    // NS_LOG_UNCOND("pair.first = "<<pair.first<<"; pair.second = "<<pair.second);
    m_link_stats_map[pair.first] = 0;
    m_link_dateRate_vector[pair.second] = 0;
    m_Tx_packetSizeSum_map[pair.first] = 0;
    m_Tx_cnt_TxPacket_map[pair.first] = 0;
  }

  // m_classifier = classifier;
  // m_monitor = monitor;
  m_monitor_interval_time = 0.1;//！
  m_prev_rxBytes = 0;
  m_prev_txBytes = 0;
  m_curTime = 0;
  m_prevTime = 0;

  m_port_onoff = 9000;   // Discard port (RFC 863)

  // link索引->link string
  for (const auto& pair : m_link_str_to_idx_map) {
    m_link_idx_to_str_vector[pair.second] = pair.first;//！需要打印检查m_link_idx_to_str_vector是否正确
    // std::cout << "pair.second \"" << pair.second << "\" => pair.first: " << pair.first << std::endl;
  }
  // for (const auto& value : m_link_idx_to_str_vector) {
  //     NS_LOG_UNCOND("cnt = " << value);
  // }

  // 从一个向量中找出包含某一节点ID的元素,并组成一个向量。例如：对于节点 2, 可以得到Device向量{"1_2","2_3","2_6"}，注意对于该节点的DeviceList对应为1,2，3
  std::vector<std::string> v_idx_ip_itf_temper;
  std::vector<std::string> v_find_temper;
  uint32_t nodeIdBefore;
  uint32_t nodeIdAfter;
  size_t underscore_pos;
  for (const auto& pair : m_devs_cont_map) {
    v_idx_ip_itf_temper.push_back(pair.first);//！需要打印检查m_link_idx_to_str_vector是否正确
  }
  for (uint32_t i = 0; i < m_n_nodes; i++) {
    for (const auto& dev_str : v_idx_ip_itf_temper) {
      underscore_pos = dev_str.find_first_of('_');
      // 提取下划线之前的子字符串
      std::string first_part = dev_str.substr(0, underscore_pos);
      std::string second_part = dev_str.substr(underscore_pos + 1);
      // 转换子字符串为uint32_t
      nodeIdBefore = std::stoi(first_part); // "_"之前的数字id
      nodeIdAfter = std::stoi(second_part); // "_"之后的数字id
      if (i == nodeIdBefore || i == nodeIdAfter) {
          v_find_temper.push_back(dev_str);
      }
    }
    std::sort(v_find_temper.begin(), v_find_temper.end(), compareStringsByNumbers); // 对向量v_find_temper中的元素排序
    m_nodeId_to_DevStr_map[i] = v_find_temper;
    v_find_temper.clear();
  }
  // for (int is=0;is<m_n_nodes;is++) {
  //     NS_LOG_UNCOND("is = " << is);
  //     for (const auto& s : m_nodeId_to_DevStr_map[is]) {
  //     NS_LOG_UNCOND("s = " << s);
  //     }
  // }


  // 遍历原始向量中的每个字符串
  for (const std::string& str : m_link_idx_to_str_vector) {
    std::size_t pos = str.find('_');
    if (pos == std::string::npos) {
      // 没有找到下划线，格式不正确
      std::cerr << "Invalid format in string: " << str << std::endl;
      continue; // 跳过当前字符串，继续下一个
    }

    std::string part1 = str.substr(0, pos);
    std::string part2 = str.substr(pos + 1);

    try {
      uint32_t value1 = std::stoul(part1);
      uint32_t value2 = std::stoul(part2);
      m_from_node_vector.push_back(value1);// 结果向量
      m_to_node_vector.push_back(value2);  // 结果向量
    } catch (const std::invalid_argument& e) {
      // 转换错误，part1或part2不是有效的无符号长整型
      std::cerr << "Invalid argument for conversion in string: " << str << std::endl;
    } catch (const std::out_of_range& e) {
      // 转换结果超出uint32_t的范围
      std::cerr << "Out of range for conversion in string: " << str << std::endl;
    }
  }

  // ####### VNR设置 #########
  m_graphVNR_vector = graphVNR_vector;

  // VNR到达时间间隔，服从柏松份分布
  Ptr<ExponentialRandomVariable> exponRandomVar = CreateObject<ExponentialRandomVariable>();
  double lambda = 0.2;//0.2;//设置柏松分布的到达率 lambda
  double exp_r_mean = 1 / lambda;//也可以直接设置指数分布的Mean
  exponRandomVar->SetAttribute ("Mean", DoubleValue (exp_r_mean));//这里的mean相当于 1/lambda
  exponRandomVar->SetAttribute ("Bound", DoubleValue (10.0));//8//设置随机输出值的上限

  for (uint32_t i = 0; i < static_cast<uint32_t>(m_graphVNR_vector.size() ); ++i) {
    m_interval_vector.push_back(exponRandomVar->GetValue() );
  }
  // VNR的生命周期，服从指数分布
  exp_r_mean = 30;// 30 //设置指数分布的Mean// ！！！
  exponRandomVar->SetAttribute ("Mean", DoubleValue (exp_r_mean));//这里的mean相当于 1/lambda
  exponRandomVar->SetAttribute ("Bound", DoubleValue (45.0));// 45 //设置随机输出值的上限// ！！！
  for (uint32_t i = 0; i < static_cast<uint32_t>(m_graphVNR_vector.size() ); ++i) {
    m_life_cycle_vector.push_back(exponRandomVar->GetValue() );
    NS_LOG_UNCOND( "m_life_cycle_vector[i] = "<<m_life_cycle_vector[i]<< "; i = " << i);
  }

  // ######## NodeStepCache 初始化 ########
  for (uint32_t i = 0; i < m_n_nodes; ++i) {
    std::map<uint32_t, NodeStepCache> m_node_step_cache_map;// 创建一个新的 m_node_step_cache_map 局部实例，
    m_node_step_cache_map = {{0, NodeStepCache(0,0,0,0,0)}};//并初始化一个条目
    m_node_cache_map[i] = m_node_step_cache_map;
  }
  // for (uint32_t i = 0; i < m_n_nodes; ++i) {
  //   std::vector<NodeStepCache> m_node_step_cache_vector;// 创建一个新的 m_node_step_cache_vector 局部实例，
  //   m_node_step_cache_vector.push_back(NodeStepCache(i, 0.0, 0.0, 0.0, 0.0) );//并初始化一个条目
  //   m_node_cache_vector[i] = m_node_step_cache_vector;
  // }

  m_cnt_VNR = 0;
  m_cnt_step = 0;
  m_cnt_step_int = -1;
  // total_number_VNR = graphVNR_vector.size();
  // ##**##
  MyGymEnv::GetNetStats (); // 启动节点挂钩，实时监控网口速率

  Simulator::Schedule (Seconds(0.0), &MyGymEnv::ScheduleNextStateRead, this);
}

void
MyGymEnv::ScheduleNextStateRead ()
{
  ///
  Time CurTime0 = Now (); //得到当前时间（Time 类型）
  double curTxTime0 = CurTime0.GetSeconds(); //得到当前时间(double 类型）
  ///
  NS_LOG_UNCOND( "==>> ***SchNext step begin*** << m_cnt_step =" << m_cnt_step<< "; m_cnt_VNR = " << m_cnt_VNR);
  // NS_LOG_FUNCTION (this);

  // ### VNR
  NS_LOG_UNCOND( "==>> *****SchNext***** << m_graphVNR_vector.size() =" << m_graphVNR_vector.size());
  NS_LOG_UNCOND( "==>> ******SchNext******* << m_cnt_VNR =" << m_cnt_VNR);

  m_graphVNR = m_graphVNR_vector.at(m_cnt_VNR);
  m_interval = m_interval_vector.at(m_cnt_VNR);
  m_life_cycle = m_life_cycle_vector.at(m_cnt_VNR);

  // ### 网络
  // ##**## 重置更新并读取当前网口速率
  // 设置Tx网口状态更新读取时间
  if (m_interval <= 2) {
    m_interval = 2;
  }
  if (m_life_cycle <= 15) {// ！！！
    m_life_cycle = 15;// ！！！
  }
  //
  double t_Tx_read; // 注意在变量命名中 t 为时间间隔，tau和Time为具体时刻
  double t_rTx = 2;
  if (m_interval - t_rTx < t_rTx){
    t_Tx_read = 0;
  } else {
    t_Tx_read = m_interval - t_rTx;
  }

  // ///
  // t_Tx_read = 47;
  // m_interval = 50;
  // m_life_cycle = 49;
  // ///
  NS_LOG_UNCOND( "==>> ******SchNext******* << m_interval =" << m_interval
  << "; m_life_cycle =" << m_life_cycle<< "; t_Tx_read =" << t_Tx_read);

  // ##**##
  Simulator::Schedule (Seconds(t_Tx_read), &MyGymEnv::ResetReadNetStats, this); //！
  Simulator::Schedule (Seconds (m_interval), &MyGymEnv::ScheduleNextStateRead, this);
  Notify();

  // ##**##
  // ### 获取Reward所需要的信息(上一个step的信息)
  if (m_cnt_step_int != -1) {
    m_rwd_graphVNR = m_graphVNR;
    m_rwd_cnt_VNR = m_cnt_VNR;
    m_rwd_interval = m_interval;
    m_rwd_life_cycle = m_life_cycle;

    m_cnt_VNR = m_cnt_VNR + 1;
    m_cnt_step = m_cnt_step + 1;

  } else { //此时MyGymEnv初始化
    m_cnt_step_int = m_cnt_step_int + 1;
    m_cnt_VNR = 0;
    m_cnt_step = 0;
  }

  if (m_log_state_fixed == true) {
    if (m_cnt_VNR >= m_state_buffer_size) {
      m_cnt_VNR = 0;
    }
  } else {
    if (m_cnt_VNR >= static_cast<uint32_t>(m_graphVNR_vector.size() ) ) {
      m_cnt_VNR = 0;
    }
  }

  NS_LOG_UNCOND( "==>> *******SchNext Step end******* m_cnt_step = " << m_cnt_step);
}

MyGymEnv::~MyGymEnv ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
MyGymEnv::GetTypeId (void)
{
  static TypeId tid = TypeId ("MyGymEnv")
    .SetParent<OpenGymEnv> ()
    .SetGroupName ("OpenGym")
    .AddConstructor<MyGymEnv> ()
  ;
  return tid;
}

void
MyGymEnv::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

void
MyGymEnv::ResetReadNetStats ()
{
  // 重置更新 m_link_stats_map, m_link_dateRate_vector, m_link_curTxTime_vector
  // => 开始读取当前网口速率
  ///
  Time CurTime4 = Now (); //得到当前时间（Time 类型）
  double curTxTime4 = CurTime4.GetSeconds(); //得到当前时间(double 类型）
  NS_LOG_UNCOND( "==>> *******get ### cResetReadNetStats ###****** curTxTime4 = " << curTxTime4);
  ///

  for (const auto& pair : m_link_str_to_idx_map) {
    // NS_LOG_UNCOND("pair.first = "<<pair.first<<"; pair.second = "<<pair.second);
    m_link_stats_map[pair.first] = 0;
    m_link_dateRate_vector[pair.second] = 0;
    m_Tx_packetSizeSum_map[pair.first] = 0;
    m_Tx_cnt_TxPacket_map[pair.first] = 0;
    m_link_curTxTime_vector[pair.second] = 0;
  }

  Time CurTime = Now (); //得到当前时间（Time 类型）
  double curTxTime = CurTime.GetSeconds(); //得到当前时间(double 类型）
  m_tau_Tx_read = curTxTime; // 开始 Tx_read 的时刻

  m_log_Tx_read = true; // Tx read begin
  // NS_LOG_UNCOND( "==>> *******get ### cResetReadNetStats ### ****** m_log_Tx_read = " << m_log_Tx_read);
}

void
MyGymEnv::TraceSinkNetStats (std::string context, Ptr<const Packet> macTxTrace)
{
  if (m_log_Tx_read == true)
  {
    // // NS_LOG_UNCOND( "==>> *******begin Tx_read ****** ");
    // // ###(0)### 获取当前的时间
    // Time CurTime = Now (); //得到当前时间（Time 类型）
    // double curTxTime = CurTime.GetSeconds(); //得到当前时间(double 类型）

    // ###(1)### 从context取出 nodeIndex 和 devIndex
    std::size_t nodeListPos = context.find("/NodeList/");
    std::size_t deviceListPos = context.find("/DeviceList/");
    // 提取"NodeList/"后面的数字
    nodeListPos += strlen("/NodeList/"); // 跳过"/NodeList/"的长度
    std::size_t nextSlashPos = context.find('/', nodeListPos); // 查找下一个'/'的位置
    std::string nodeListNumberStr = context.substr(nodeListPos, nextSlashPos - nodeListPos);
    // 提取"DeviceList/"后面的数字
    deviceListPos += strlen("/DeviceList/"); // 跳过"/DeviceList/"的长度
    nextSlashPos = context.find('/', deviceListPos); // 查找下一个'/'的位置
    std::string deviceListNumberStr = context.substr(deviceListPos, nextSlashPos - deviceListPos);
    // 将字符串转换为整数
    uint32_t nodeIndex = std::stoi(nodeListNumberStr);
    uint32_t devIndex = std::stoi(deviceListNumberStr);
    // 输出结果
    // std::cout << "NodeList number: " << nodeIndex << std::endl;
    // std::cout << "DeviceList number: " << devIndex << std::endl;

    // ###(2)### 获取link_idx
    std::vector<std::string> nd_dev_str_vector = m_nodeId_to_DevStr_map[nodeIndex];
    std::string dev_str = nd_dev_str_vector[devIndex-1];
    uint32_t nodeIdBefore;
    uint32_t nodeIdAfter;
    size_t underscore_pos = dev_str.find_first_of('_');
    // 提取下划线之前的子字符串
    std::string first_part = dev_str.substr(0, underscore_pos);
    std::string second_part = dev_str.substr(underscore_pos + 1);
    // 转换子字符串为uint32_t
    nodeIdBefore = std::stoi(first_part); // "_"之前的数字id
    nodeIdAfter = std::stoi(second_part); // "_"之后的数字id
    std::string link_str;
    if (nodeIdBefore == nodeIndex) {
      link_str = dev_str; // std::to_string(nodeIndex) + std::string("_") + std::to_string(nodeIdAfter);
    } else {
      link_str = std::to_string(nodeIdAfter) + std::string("_") + std::to_string(nodeIdBefore);
    }
    uint32_t link_idx = m_link_str_to_idx_map[link_str];

    // ###(3)### 更新Tx的网口速率
    uint32_t packetSizeSum = m_Tx_packetSizeSum_map[link_str];
    // 获得发包的大小
    // MacTx挂钩：Trace source indicating a packet has arrived for transmission by this device.
    uint32_t packet_size = macTxTrace->GetSize();//单位 byte
    uint32_t cnt_TxPacket = m_Tx_cnt_TxPacket_map[link_str]; // 0 为首次发包
    if (cnt_TxPacket == 0) {
      packet_size = static_cast<uint32_t>(static_cast<double>(packet_size) / 2); // 为了减少计算网口速率时的误差
    }
    packetSizeSum = packetSizeSum + packet_size;
    // NS_LOG_UNCOND (context << " sizePacket = " << packet_size);//！

    // // ### 计算Tx的网口速率 bps (bite/1s)//注意 packet_size并不是 clientHelper.SetAttribute ："PacketSize"中设置的数值
    // // UDP数据包会在每个设置的数值为基础，加上30byte的UDP包头，例如 "PacketSize"设置为80,则实际的packet_size为 80+30=110
    // double prevTxTime = m_link_curTxTime_vector[link_idx];
    // double dateRate = 8 * static_cast<double>(packetSizeSum) / (curTxTime - m_tau_Tx_read); // 注意在变量命名时，tau与Time均为时刻
    // double dateRate1 = 8 * packet_size / (curTxTime - prevTxTime);

    // NS_LOG_UNCOND ("curTxTime = "<< curTxTime <<"; prevTxTime = "<< prevTxTime << "; dateRate1 = " << dateRate1);//！打印显示

    // NS_LOG_UNCOND ("cnt_TxPacket = "<< cnt_TxPacket <<"; nodeIndex = " << nodeIndex << "; link_str = " << link_str
    //                <<"; packet_size = "<< packet_size << "curTxTime = "<< curTxTime << "; dateRate = " << dateRate);//！打印显示

    // 更新
    // m_link_stats_map[link_str] = dateRate;
    // m_link_dateRate_vector[link_idx] = dateRate;
    // m_link_curTxTime_vector[link_idx] = curTxTime;
    m_Tx_packetSizeSum_map[link_str] = packetSizeSum;
    m_Tx_cnt_TxPacket_map[link_str] = cnt_TxPacket + 1;
  }
}

void
MyGymEnv::GetNetStats ()
{
// m_link_stats_map, key为link_str, value为link当前状态信息：链路负载（link已用带宽）
// 即：将link发送端的网口发包速率（Tx,单位 bit/1s (bps)）作为当前link已用带宽
// m_link_dateRate_vector，索引为link idx，value为link发送端的网口发包速率（Tx,单位 bit/1s (bps)
  uint32_t nodeIndex; //node id: 0,1,2,3,..
  uint32_t devIndex; // 注意NetDevice 0 不可用，它为LookbackNetDevice；
                     // NetDevice 1,2,.. 可用，该节点DeviceList的Id按照IP地址由低到高排序

  for (const auto& link_str : m_link_idx_to_str_vector) {
    // ###步骤(1)### 获取 nodeIndex
    // 将字符串中"_"前的字符取出并转为uint32_t类型，例如 "12_6" => 12
    uint32_t nodeIdAfter;
    size_t underscore_pos = link_str.find_first_of('_');
    if (underscore_pos != std::string::npos) {
        // 提取下划线之前的子字符串
        std::string first_part = link_str.substr(0, underscore_pos);
        std::string second_part = link_str.substr(underscore_pos + 1);
        try {
            // 转换子字符串为uint32_t
            nodeIndex = std::stoi(first_part); // "_"之前的数字id
            nodeIdAfter = std::stoi(second_part); // "_"之后的数字id
            // std::cout << "Number before underscore: " << nodeIndex << std::endl;

        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << e.what() << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "No underscore found in the string." << std::endl;
    }

    // ###步骤(2)### 获取 devIndex
    // 例如 对于节点 2, 可以得到Device向量{"1_2","2_3","2_6"}，注意对于该节点的DeviceList对应为1, 2，3
    std::string dev_str;
    if (nodeIndex < nodeIdAfter) {
      dev_str = link_str; // std::to_string(nodeIndex) + std::string("_") + std::to_string(nodeIdAfter);
    } else {
      dev_str = std::to_string(nodeIdAfter) + std::string("_") + std::to_string(nodeIndex);
    }

    std::vector<std::string> nd_dev_str_vector = m_nodeId_to_DevStr_map[nodeIndex];
    for (const auto& value : nd_dev_str_vector) {
      NS_LOG_UNCOND("value = " << value);
    }

    // 使用std::find查找字符串在向量中的位置
    auto it_pos = std::find(nd_dev_str_vector.begin(), nd_dev_str_vector.end(), dev_str);
    // 检查是否找到了目标字符串
    if (it_pos != nd_dev_str_vector.end()) {
        // 计算索引位置，并转换为 uint32_t 类型
        devIndex = 1 + static_cast<uint32_t>(std::distance(nd_dev_str_vector.begin(), it_pos));
    } else {
        std::cout << "String \"" << dev_str << "\" not found in the nd_dev_str_vector." << std::endl;
    }

    // ###步骤(3)### 得到各个链路的网口速率
    std::ostringstream oss;
    NS_LOG_UNCOND( "==>> ***********link_str = "<<link_str<<"; dev_str = "<<dev_str);
    oss << "/NodeList/"<< nodeIndex << "/DeviceList/" << devIndex << "/$ns3::PointToPointNetDevice/MacTx";
    NS_LOG_UNCOND( "==>> *********** "<<oss.str());
    Config::Connect(oss.str(), MakeCallback(&MyGymEnv::TraceSinkNetStats, this));
  }
}

/*
Define observation space
*/
Ptr<OpenGymSpace>
MyGymEnv::GetObservationSpace()
{
  // ########## VNR #########

  // VNR为了固定状态空间，以最大虚拟节点数目和最大链路数目进行设置，因此具体的VNR状态数据需要补零

  // box_VNR_1
  // m_n_nodes_VNR = m_graphVNR.getNodeNumber();
  // m_n_links_VNR = m_graphVNR.getEdgeNumber();

  float low_VNR_1 = 0.0;
  float high_VNR_1 = static_cast<float>(std::max(m_n_max_VNF, m_n_max_links_VNR)); //*//一个VNR的虚拟节点和link数目的最大值（VNR_number_node_feature和VNR_number_link_feature也不超过该值）
  std::vector<uint32_t> shape_VNR_1 = {4 + 2 * m_n_max_links_VNR,};
  std::string dtype_VNR_1 = TypeNameGet<double> ();
  Ptr<OpenGymBoxSpace> box_VNR_1 = CreateObject<OpenGymBoxSpace> (low_VNR_1, high_VNR_1, shape_VNR_1, dtype_VNR_1);

  // box_VNR_2
  float low_VNR_2 = 0.0;
  float high_VNR_2 = 10.0; //1 //*//节点资源的最大值 10
  std::vector<uint32_t> shape_VNR_2 = {m_n_max_VNF * 2,};//！！节点特征的属性有2个
  std::string dtype_VNR_2 = TypeNameGet<double> ();
  Ptr<OpenGymBoxSpace> box_VNR_2 = CreateObject<OpenGymBoxSpace> (low_VNR_2, high_VNR_2, shape_VNR_2, dtype_VNR_2);

  // box_VNR_3
  float low_VNR_3 = 0.0;
  float high_VNR_3 = 10.0; // 1 //*//10.0是因为GetObservation()的box_VNR_3进行了 *10
  std::vector<uint32_t> shape_VNR_3 = {m_n_max_links_VNR,};
  std::string dtype_VNR_3 = TypeNameGet<double> ();
  Ptr<OpenGymBoxSpace> box_VNR_3 = CreateObject<OpenGymBoxSpace> (low_VNR_3, high_VNR_3, shape_VNR_3, dtype_VNR_3);

  // box_VNR_VNF
  float low_VNR_VNF = 0.0;
  float high_VNR_VNF = static_cast<float>(m_n_type_VNF - 1);
  std::vector<uint32_t> shape_VNR_VNF = {m_n_max_VNF,};
  std::string dtype_VNR_VNF = TypeNameGet<double> ();
  Ptr<OpenGymBoxSpace> box_VNR_VNF = CreateObject<OpenGymBoxSpace> (low_VNR_VNF, high_VNR_VNF, shape_VNR_VNF, dtype_VNR_VNF);


  // // box_VNR_1
  // m_n_nodes_VNR = m_graphVNR.getNodeNumber();
  // m_n_links_VNR = m_graphVNR.getEdgeNumber();
  // float low_VNR_1 = 0.0;
  // float high_VNR_1 = static_cast<float>(m_n_max_VNF - 1); //*//一个VNR的最大虚拟节点数目（VNR_number_node_feature和VNR_number_link_feature也不超过该值）
  // std::vector<uint32_t> shape_VNR_1 = {4 + 2 * m_n_links_VNR,};
  // std::string dtype_VNR_1 = TypeNameGet<double> ();
  // Ptr<OpenGymBoxSpace> box_VNR_1 = CreateObject<OpenGymBoxSpace> (low_VNR_1, high_VNR_1, shape_VNR_1, dtype_VNR_1);

  // // box_VNR_2
  // float low_VNR_2 = 0.0;
  // float high_VNR_2 = 10.0; //1 //*//节点资源的最大值 10
  // std::vector<uint32_t> shape_VNR_2 = {m_n_nodes_VNR * 2,};//！！节点特征的属性有2个
  // std::string dtype_VNR_2 = TypeNameGet<double> ();
  // Ptr<OpenGymBoxSpace> box_VNR_2 = CreateObject<OpenGymBoxSpace> (low_VNR_2, high_VNR_2, shape_VNR_2, dtype_VNR_2);

  // // box_VNR_3
  // float low_VNR_3 = 0.0;
  // float high_VNR_3 = 10.0; // 1 //*//10.0是因为GetObservation()的box_VNR_3进行了 *10
  // std::vector<uint32_t> shape_VNR_3 = {m_n_links_VNR,};
  // std::string dtype_VNR_3 = TypeNameGet<double> ();
  // Ptr<OpenGymBoxSpace> box_VNR_3 = CreateObject<OpenGymBoxSpace> (low_VNR_3, high_VNR_3, shape_VNR_3, dtype_VNR_3);

  // // box_VNR_VNF
  // float low_VNR_VNF = 0.0;
  // float high_VNR_VNF = static_cast<float>(m_n_type_VNF - 1);
  // std::vector<uint32_t> shape_VNR_VNF = {m_n_nodes_VNR,};
  // std::string dtype_VNR_VNF = TypeNameGet<double> ();
  // Ptr<OpenGymBoxSpace> box_VNR_VNF = CreateObject<OpenGymBoxSpace> (low_VNR_VNF, high_VNR_VNF, shape_VNR_VNF, dtype_VNR_VNF);


  // ######## Network #######
  // box_net_1
  float low_net_1 = 0.0;
  float high_net_1 = static_cast<float>(std::max(m_n_nodes, m_n_links)); //*//net的节点m_n_nodes与m_n_links的最大值（Vnumber_node_feature 和 number_link_feature,也不超过该值）
  std::vector<uint32_t> shape_net_1 = {4 + 2 * m_n_links,};
  std::string dtype_net_1 = TypeNameGet<double> ();
  Ptr<OpenGymBoxSpace> box_net_1 = CreateObject<OpenGymBoxSpace> (low_net_1, high_net_1, shape_net_1, dtype_net_1);

  // box_net_2
  float low_net_2 = 0.0;
  float high_net_2 = 1.0;
  std::vector<uint32_t> shape_net_2 = {m_n_nodes * 4,};//！！节点特征的属性有4个
  std::string dtype_net_2 = TypeNameGet<double> ();
  Ptr<OpenGymBoxSpace> box_net_2 = CreateObject<OpenGymBoxSpace> (low_net_2, high_net_2, shape_net_2, dtype_net_2);

  // box_net_3
  float low_net_3 = 0.0;
  float high_net_3 = 10.0; //！！
  std::vector<uint32_t> shape_net_3 = {m_n_links,};
  std::string dtype_net_3 = TypeNameGet<double> ();
  Ptr<OpenGymBoxSpace> box_net_3 = CreateObject<OpenGymBoxSpace> (low_net_3, high_net_3, shape_net_3, dtype_net_3);

  // box_net_VNF
  float low_net_VNF = 0.0;
  float high_net_VNF = 1.0;
  std::vector<uint32_t> shape_net_VNF = {m_n_nodes * m_n_type_VNF};
  std::string dtype_net_VNF = TypeNameGet<double> ();
  Ptr<OpenGymBoxSpace> box_net_VNF = CreateObject<OpenGymBoxSpace> (low_net_VNF, high_net_VNF, shape_net_VNF, dtype_net_VNF);

  // ########################
  Ptr<OpenGymDictSpace> space = CreateObject<OpenGymDictSpace> ();

  space->Add("box_VNR_1",box_VNR_1);
  space->Add("box_VNR_2",box_VNR_2);
  space->Add("box_VNR_3",box_VNR_3);
  space->Add("box_VNR_VNF",box_VNR_VNF);

  space->Add("box_net_1",box_net_1);
  space->Add("box_net_2",box_net_2);
  space->Add("box_net_3",box_net_3);
  space->Add("box_net_VNF",box_net_VNF);

  NS_LOG_UNCOND ("MyGetObservationSpace: " << space);
  return space;
}

/*
Define action space
*/
Ptr<OpenGymSpace>
MyGymEnv::GetActionSpace()
{

  float low_action = 0.0;
  float high_action = static_cast<float>(m_n_nodes - 1);
  std::vector<uint32_t> shape_action = {m_n_max_VNF,};
  std::string dtype_action = TypeNameGet<uint32_t> ();
  Ptr<OpenGymBoxSpace> box_action = CreateObject<OpenGymBoxSpace> (low_action, high_action, shape_action, dtype_action);
  return box_action;

  // Ptr<OpenGymDiscreteSpace> discrete = CreateObject<OpenGymDiscreteSpace> (m_n_max_VNF);
  // return discrete;

}

/*
Define game over condition
*/
bool
MyGymEnv::GetGameOver()
{
  bool isGameOver = false;
  // bool test = false;
  bool test = true;
  static uint32_t stepCounter = 0;
  // stepCounter += 1;
  stepCounter = m_cnt_step;
  if (stepCounter == m_stepTotalNumber && test) {
      isGameOver = true;
  }
  NS_LOG_UNCOND ("MyGetGameOver: " << isGameOver << "; m_stepTotalNumber = "<<m_stepTotalNumber<<"; stepCounter = "<<stepCounter);
  return isGameOver;
}

/*
Collect observations
*/
Ptr<OpenGymDataContainer>
MyGymEnv::GetObservation()
{
  // ########## VNR #########
  // box_VNR_1，拓扑数据，向量数据格式:{n_nodes_VNR, VNR_number_node_feature, n_links_VNR, VNR_number_link_feature,
  //                                VNR_from_nodes索引(...), VNR_to_nodes索引(...), }
  // box_VNR_2，节点特征数据，向量数据格式:{节点0的特征向量,节点1的特征向量,..,..}
  // box_VNR_3，边特征数据，向量数据格式:{按边的索引排序(...)} //每个边的特征仅有一个属性
  // box_VNR_VNF，在一个VNR中，每个虚拟节点所请求的VNF类型，（uint32_t, 0 1 2 3 4）
  // 其中 box_VNR_1 和 box_VNR_3 均为双向边的数据

  // box_VNR_1
  m_n_nodes_VNR = m_graphVNR.getNodeNumber();
  m_n_links_VNR = m_graphVNR.getEdgeNumber();
  std::vector<uint32_t> shape_VNR_1 = {4 + 2 * m_n_links_VNR,};
  Ptr<OpenGymBoxContainer<double> > box_VNR_1 = CreateObject<OpenGymBoxContainer<double> >(shape_VNR_1);
  box_VNR_1->AddValue(static_cast<double>(m_n_nodes_VNR));
  uint32_t VNR_number_node_feature = 2; //！！VNR节点特征的属性有2个 // 注意这里除以10是为了不超出[0,1]的box范围
  box_VNR_1->AddValue(static_cast<double>(VNR_number_node_feature));

  box_VNR_1->AddValue(static_cast<double>(m_n_links_VNR));
  uint32_t VNR_number_link_feature = 1; //！！链路特征的属性有1个 // 注意这里除以10是为了不超出[0,1]的box范围
  box_VNR_1->AddValue(static_cast<double>(VNR_number_link_feature));

  auto [VNR_from_node_vector, VNR_to_node_vector] = m_graphVNR.getFromToNodeIndex(); // C++17结构化绑定
  for (const auto& value : VNR_from_node_vector) {
    box_VNR_1->AddValue(static_cast<double>(value));
  }
  for (const auto& value : VNR_to_node_vector) {
    box_VNR_1->AddValue(static_cast<double>(value));
  }
  // ### 补零 ###
  if (m_n_max_links_VNR - m_n_links_VNR > 0){
    std::vector<double> v_z_VNR_1(static_cast<uint32_t>(static_cast<double>(m_n_max_links_VNR - m_n_links_VNR) * 2), 0.0);
    for (const auto& value : v_z_VNR_1) {
      box_VNR_1->AddValue(value);
    }
  }

  // box_VNR_2
  std::vector<uint32_t> shape_VNR_2 = {m_n_nodes_VNR * VNR_number_node_feature,};
  // Ptr<OpenGymBoxContainer<std::vector<double> > > box_VNR_2 = CreateObject<OpenGymBoxContainer<std::vector<double>> >(shape_VNR_2);
  Ptr<OpenGymBoxContainer<double> > box_VNR_2 = CreateObject<OpenGymBoxContainer<double> >(shape_VNR_2);
  AttrNodeVNR node_attr_VNR;
  for (const auto& pair : m_graphVNR.adjList) {
    // node_attr_VNR = pair.second.node_attribute;
    node_attr_VNR = m_graphVNR.getNodeAttribute(pair.first);
    box_VNR_2->AddValue(node_attr_VNR.d_R1); // 未归一化
    box_VNR_2->AddValue(node_attr_VNR.d_R2); // 未归一化
    // std::vector<double> VNR_v_node_feature = {node_attr_VNR.d_R1, node_attr_VNR.d_R2};
    // box_VNR_2->AddValue(VNR_v_node_feature);
  }
  // ### 补零 ###
  if (m_n_max_VNF - m_n_nodes_VNR > 0){
    std::vector<double> v_z_VNR_2(static_cast<uint32_t>(static_cast<double>(m_n_max_VNF - m_n_nodes_VNR) * 2), 0.0);
    for (const auto& value : v_z_VNR_2) {
      box_VNR_2->AddValue(value);
    }
  }

  // box_VNR_3
  std::vector<uint32_t> shape_VNR_3 = {m_n_links_VNR,};
  Ptr<OpenGymBoxContainer<double> > box_VNR_3 = CreateObject<OpenGymBoxContainer<double> >(shape_VNR_3);
  std::map<uint32_t, double> to_id_attr_map;
  for (const auto& pair : m_graphVNR.adjList) {
    for (const auto& value : pair.second.outEdges) {
      to_id_attr_map[value.target] = value.edge_attribute.b_e;
    }
    for (const auto& pair_attr : to_id_attr_map) {
      box_VNR_3->AddValue(pair_attr.second / (100*1000) * 10);//100*1000为物理链路带宽总量, *10 有待考虑，是为了提升该带宽请求的数值
    }
    to_id_attr_map.clear();
  }
  // ### 补零 ###
  if (m_n_max_links_VNR - m_n_links_VNR > 0){
    std::vector<double> v_z_VNR_3(m_n_max_links_VNR - m_n_links_VNR, 0.0);
    for (const auto& value : v_z_VNR_3) {
      box_VNR_3->AddValue(value);
    }
  }


  // box_VNR_VNF
  std::vector<uint32_t> shape_VNR_VNF = {m_n_nodes_VNR,};
  Ptr<OpenGymBoxContainer<double> > box_VNR_VNF = CreateObject<OpenGymBoxContainer<double> >(shape_VNR_VNF);
  for (const auto& pair : m_graphVNR.adjList) {
    // node_attr_VNR = pair.second.node_attribute;
    node_attr_VNR = m_graphVNR.getNodeAttribute(pair.first);
    box_VNR_VNF->AddValue(static_cast<double>(node_attr_VNR.type_VNF));
  }
  // ### 补零 ###
  if (m_n_max_VNF - m_n_nodes_VNR > 0){
    std::vector<double> v_z_VNR_VNF(m_n_max_VNF - m_n_nodes_VNR, 0.0);
    for (const auto& value : v_z_VNR_VNF) {
      box_VNR_VNF->AddValue(value);
    }
  }

  // ######## Network #######
  // box_net_1，拓扑数据，向量数据格式:{n_nodes, number_node_feature, n_links, number_link_feature,
  //                                from_nodes索引(...), to_nodes索引(...), }
  // box_net_2，节点特征数据，向量数据格式:{节点0的特征向量,节点1的特征向量,..,..}
  // box_net_3，边特征数据，向量数据格式:{按边的索引排序(...)} //每个边的特征仅有一个属性：链路利用（负载）率
  // box_net_VNF, 在网络中各个网络节点所部署的VNF信息，向量数据格式:{0,1,0,0,0, 0,1,0,1,0, 0,0,0,1,0, ..}
  // 其中 box_net_1 和 box_net_3 均为双向边的数据

  // box_net_1
  std::vector<uint32_t> shape_net_1 = {4 + 2 * m_n_links,};
  Ptr<OpenGymBoxContainer<double> > box_net_1 = CreateObject<OpenGymBoxContainer<double> >(shape_net_1);
  box_net_1->AddValue(static_cast<double>(m_n_nodes));
  uint32_t number_node_feature = 4; //！！节点特征的属性有4个 // 注意这里除以10是为了不超出[0,1]的box范围
  box_net_1->AddValue(static_cast<double>(number_node_feature));

  box_net_1->AddValue(static_cast<double>(m_n_links));
  uint32_t number_link_feature = 1; //！！链路特征的属性有1个 // 注意这里除以10是为了不超出[0,1]的box范围
  box_net_1->AddValue(static_cast<double>(number_link_feature));

  for (const auto& value : m_from_node_vector) {
    box_net_1->AddValue(static_cast<double>(value));
  }
  for (const auto& value : m_to_node_vector) {
    box_net_1->AddValue(static_cast<double>(value));
  }

  // box_net_2
  std::vector<uint32_t> shape_net_2 = {m_n_nodes * number_node_feature,};
  // Ptr<OpenGymBoxContainer<std::vector<double> > > box_net_2 = CreateObject<OpenGymBoxContainer<std::vector<double>> >(shape_net_2);
  Ptr<OpenGymBoxContainer<double> > box_net_2 = CreateObject<OpenGymBoxContainer<double> >(shape_net_2);
  for (const auto& value : m_node_attr_vector) {
    // std::vector<double> v_node_feature = {value.D_R1_res / value.D_R1, value.D_R1_unit_cost,
    //                                       value.D_R2_res / value.D_R2, value.D_R2_unit_cost};
    // box_net_2->AddValue(v_node_feature);
    box_net_2->AddValue(6 * (value.D_R1 - value.D_R1_res) / value.D_R1);
    box_net_2->AddValue(value.D_R1_unit_cost);
    box_net_2->AddValue(6 * (value.D_R2 - value.D_R2_res) / value.D_R2);
    box_net_2->AddValue(value.D_R2_unit_cost);
  }

  // box_net_3
  std::vector<uint32_t> shape_net_3 = {m_n_links,};
  Ptr<OpenGymBoxContainer<double> > box_net_3 = CreateObject<OpenGymBoxContainer<double> >(shape_net_3);
  std::vector<LinkAttr> link_attr_vector(m_n_links);

  // 计算获取link的网口Tx速率
  m_log_Tx_read = false; // 停止 Tx_read
  Time CurTime = Now (); //得到当前时间（Time 类型）
  double curTxTime = CurTime.GetSeconds(); //得到当前时间(double 类型）
  double dateRate;
  uint32_t packetSizeSum;
  for (const auto& pair : m_link_str_to_idx_map) {
    packetSizeSum = m_Tx_packetSizeSum_map[pair.first];
    if (packetSizeSum == 0) {
      dateRate = 0;
    } else {
      dateRate = 8 * static_cast<double>(packetSizeSum) / (curTxTime - m_tau_Tx_read); // 注意在变量命名时，tau与Time均为时刻
    }
    m_link_stats_map[pair.first] = dateRate;
    m_link_dateRate_vector[pair.second] = dateRate;
    // NS_LOG_UNCOND("pair.first = "<<pair.first<<"; pair.second = "<<pair.second << "; packetSizeSum = " << packetSizeSum
    // << "; curTxTime = "<<curTxTime << "; m_tau_Tx_read = "<<m_tau_Tx_read<<"; dateRate = "<<dateRate);
  }
  //更新 m_link_attr_map
  for (auto& pair : m_link_attr_map) {
    // std::string key = pair.first;
    // int idx_link = m_link_str_to_idx_map[key];
    // // LinkAttr value = pair.second;
    // link_attr_vector[idx_link] = pair.second;
    pair.second.B_res = pair.second.B - m_link_stats_map[pair.first];
    link_attr_vector[m_link_str_to_idx_map[pair.first]] = pair.second;//将边特征按照边的索引（0,1,2,..）位置存储到向量
  }
  uint32_t link_idx = 0;
  for (auto& value : link_attr_vector) {
    // value.B_res = value.B - m_link_stats_map[m_link_idx_to_str_vector[link_idx]];
    box_net_3->AddValue((value.B - value.B_res) / value.B); //link利用率
    ++link_idx;
  }

  // box_net_VNF
  std::vector<uint32_t> shape_net_VNF = {m_n_nodes * m_n_type_VNF,};
  // Ptr<OpenGymBoxContainer<std::vector<uint32_t> > > box_net_VNF = CreateObject<OpenGymBoxContainer<std::vector<uint32_t> > >(shape_net_VNF);
  Ptr<OpenGymBoxContainer<double> > box_net_VNF = CreateObject<OpenGymBoxContainer<double> >(shape_net_VNF);
  for (const auto& value_vector : m_node_VNF_d_vector) {
    for (const auto& value : value_vector) {
      box_net_VNF->AddValue(static_cast<double>(value));
    }
  }

  // ##############################
  Ptr<OpenGymDictContainer> data = CreateObject<OpenGymDictContainer> ();

  data->Add("box_VNR_1",box_VNR_1);
  data->Add("box_VNR_2",box_VNR_2);
  data->Add("box_VNR_3",box_VNR_3);
  data->Add("box_VNR_VNF",box_VNR_VNF);

  data->Add("box_net_1",box_net_1);
  data->Add("box_net_2",box_net_2);
  data->Add("box_net_3",box_net_3);
  data->Add("box_net_VNF",box_net_VNF);

  // // Print data from tuple
  // Ptr<OpenGymBoxContainer<uint32_t> > mbox = DynamicCast<OpenGymBoxContainer<uint32_t> >(data->Get("myVector"));
  // Ptr<OpenGymDiscreteContainer> mdiscrete = DynamicCast<OpenGymDiscreteContainer>(data->Get("myValue"));
  // NS_LOG_UNCOND ("MyGetObservation: " << data);
  // NS_LOG_UNCOND ("---" << mbox);
  // NS_LOG_UNCOND ("---" << mdiscrete);

  return data;
}

/*
Define reward function
*/
float
MyGymEnv::GetReward()
{
  double reward = 0; // 赋予初值
  double reward_scaling; //对reward进行整形
  double log_acc = 1; //VNR接受为1，不接受为0
  uint32_t T_step_roll = 200; // 计算长期Acc、收益、Cost等的时间长度 T

  double m_acc_rate; // 平均（T）VNR接受率
  double Rev = 0; // 该VNR的嵌入收益（Revenue）
  double Cost = 0; // 该VNR的嵌入代价（Cost）
  double revenue_cost_ratio; // 该VNR的收益代价比
  double m_L_A_Rev = 0; // 长期平均收益
  double m_L_A_Cost = 0; // 长期平均Cost
  double m_L_A_revenue_cost_ratio; // 长期平均收益代价比
  double max_link_util_rate; // 最大链路利用率
  double max_node_R1_util_rate; // 最大节点资源（R1）利用率
  double max_node_R2_util_rate; // 最大节点资源（R2）利用率
    // m_rwd_graphVNR;
    // m_rwd_cnt_VNR;
    // m_rwd_interval;
    // m_rwd_life_cycle;
  if (m_cnt_step ==0){
    NS_LOG_UNCOND( "==>> *******reward*****m_cnt_step****** " << m_cnt_step);
    return 0;
  }
  NS_LOG_UNCOND ("m_cnt_step = " << m_cnt_step << "; m_rwd_cnt_VNR = " << m_rwd_cnt_VNR);

  // 节点放置：判断节点 id_node_net 是否部署了所请求的VNF，判断剩余节点资源量是否满足所请求的资源量
  if (m_log_typeVNF == false || m_log_nodeRes == false ) {
    NS_LOG_UNCOND ("m_log_typeVNF = " << m_log_typeVNF << "; m_log_nodeRes = " << m_log_nodeRes);
    std::cout << "m_log_typeVNF = " << m_log_typeVNF << "; m_log_nodeRes = " << m_log_nodeRes << std::endl;
    log_acc = 0;
  }

  // link放置：判断link资源量是否满足所请求的资源量，为了能够评价注意不论是否满足，实验中均对节点成功放置的link进行了放置
  std::vector<double> link_B_res_vector;
  for (const auto& pair : m_link_attr_map) {
    // std::string key = pair.first;
    // int idx_link = m_link_str_to_idx_map[key];
    // // LinkAttr value = pair.second;
    // link_attr_vector[idx_link] = pair.second;

    //
    double TxdataRate = m_link_stats_map[pair.first];
    // // NS_LOG_UNCOND ("pair.first = " << pair.first << "; TxdataRate = " << TxdataRate
    // //                 <<"; pair.second.B_res = "<< pair.second.B_res<< "; pair.second.B = "<< pair.second.B);

    // if (pair.second.B_res < 0) { //！！注意如果没有考虑带宽放置失败的情况应注释掉
    //   log_acc = 0;
    //   NS_LOG_UNCOND ("log_acc = " << log_acc << "; B_res = " << pair.second.B_res);
    // }

    // // pair.second.B_res = pair.second.B - m_link_stats_map[pair.first];
    // // link_attr_vector[m_link_str_to_idx_map[pair.first]] = pair.second;//将边特征按照边的索引（0,1,2,..）位置存储到向量
  }

  // 服务时间暂没有考虑

  // ###(1)### 请求接受率 double m_acc_rate;
  m_acc_qeque.push_back(log_acc);
  if (m_acc_qeque.size() > T_step_roll) {
    m_acc_qeque.front();
  }
  double acc_num = 0;
  for (const auto& elem : m_acc_qeque) {
      acc_num = acc_num + elem;
  }
  m_acc_rate = acc_num / static_cast<double> (m_acc_qeque.size());
  std::cout << "m_acc_rate: " << m_acc_rate << std::endl;

  // ###(2)### 单个VNR的嵌入收益 Rev
  double Rev_N = 0;
  // VNR节点嵌入 Rev_N
  AttrNodeVNR node_attr_VNR;
  for (const auto& pair : m_rwd_graphVNR.adjList) {
    // node_attr_VNR = pair.second.node_attribute;
    node_attr_VNR = m_rwd_graphVNR.getNodeAttribute(pair.first);
    Rev_N = Rev_N + node_attr_VNR.d_R1 + node_attr_VNR.d_R2;
  }
  // VNR链路嵌入收益 Rev_E
  double Rev_E = 0;
  std::map<uint32_t, double> to_id_attr_map;
  for (const auto& pair : m_rwd_graphVNR.adjList) {
    for (const auto& value : pair.second.outEdges) {
      to_id_attr_map[value.target] = value.edge_attribute.b_e;
    }
    for (const auto& pair_attr : to_id_attr_map) {
      Rev_E = Rev_E + pair_attr.second;
    }
    to_id_attr_map.clear();
  }
  Rev = Rev_N + Rev_E;

  // ###(3)### 单个VNR的嵌入代价 Cost
  // VNR节点嵌入代价 Cost_N
  double Cost_N = 0;
  uint32_t rwd_n_nodes_VNR = m_rwd_graphVNR.getNodeNumber();
  std::vector<uint32_t> id_node_VNR_vector;
  for (const auto& pair : m_rwd_graphVNR.adjList) {
    id_node_VNR_vector.push_back(pair.first);
  }
  uint32_t id_node_VNR;
  uint32_t id_node_net;
  for (uint32_t i = 0; i < rwd_n_nodes_VNR - 1; ++i) {
    id_node_VNR = id_node_VNR_vector[i]; // VNR中当前节点id
    id_node_net = m_action_vector[i]; // 将VNR中当前节点放置（嵌入）至物理网络的节点id
    node_attr_VNR = m_rwd_graphVNR.getNodeAttribute(id_node_VNR);

    Cost_N = Cost_N + node_attr_VNR.d_R1 * m_node_attr_vector[id_node_net].D_R1_unit_cost
            + node_attr_VNR.d_R2 * m_node_attr_vector[id_node_net].D_R2_unit_cost;
  }

  // VNR节点嵌入代价 Cost_E
  double Cost_E = 0;
  for (const auto& pair : m_rwd_graphVNR.adjList) {
    for (const auto& value : pair.second.outEdges) {
      to_id_attr_map[value.target] = value.edge_attribute.b_e;
    }
    for (const auto& pair_attr : to_id_attr_map) {
      Cost_E = Cost_E + pair_attr.second * m_link_attr_map[m_link_idx_to_str_vector[0]].B_unit_cost;
      // ！！注意这里暂认为每个link的B_unit_cost都是相同的
    }
    to_id_attr_map.clear();
  }
  Cost = Cost_N + Cost_E;

  // ###(4)### 单个VNR的收益代价比 （Revenue-cost ratio）revenue_cost_ratio
  revenue_cost_ratio = Rev / Cost;

  // ###(5)### 长期平均收益（Long-term average revenue）m_L_A_Rev
  m_Re_qeque.push_back(Rev_N);//！！
  if (m_Re_qeque.size() > T_step_roll) {
    m_Re_qeque.front();
  }
  for (const auto& elem : m_Re_qeque) {
      m_L_A_Rev = m_L_A_Rev + elem;
  }
  m_L_A_Rev = m_L_A_Rev / static_cast<double> (m_Re_qeque.size());
  std::cout << "====>>> m_L_A_Rev_N = " << m_L_A_Rev << std::endl;

  // ###(6)### 长期平均Cost m_L_A_Cost
  m_Cost_qeque.push_back(Cost_N);//！！
  if (m_Cost_qeque.size() > T_step_roll) {
    m_Cost_qeque.front();
  }
  for (const auto& elem : m_Cost_qeque) {
      m_L_A_Cost = m_L_A_Cost + elem;
  }
  m_L_A_Cost = m_L_A_Cost / static_cast<double> (m_Cost_qeque.size());
  double rate_Rev_Cost_N = m_L_A_Rev / m_L_A_Cost;
  std::cout << "====>>> m_L_A_Cost_N = " << m_L_A_Cost << "; rate_Rev_Cost_N = " << rate_Rev_Cost_N << std::endl;

  // ###(7)### 长期平均收益代价比 m_L_A_revenue_cost_ratio
  m_L_A_revenue_cost_ratio = m_L_A_Rev / m_L_A_Cost;

  // ###(8)### 最大链路利用率 max_link_util_rate
  std::vector<double> link_util_rate_vector(m_n_links);
  std::vector<LinkAttr> link_attr_vector(m_n_links);
  for (const auto& pair : m_link_attr_map) {
    link_attr_vector[m_link_str_to_idx_map[pair.first]] = pair.second; // 将边特征按照边的索引（0,1,2,..）位置存储到向量
  }
  uint32_t link_idx = 0;
  for (const auto& value : link_attr_vector) {
    link_util_rate_vector.push_back((value.B - value.B_res) / value.B);
    ++link_idx;
  }
  auto max_lur = std::max_element(link_util_rate_vector.begin(), link_util_rate_vector.end());
  max_link_util_rate = *max_lur;


  // ###(9)### 负载均衡
  // 最大节点资源（资源类型R1和R2）的利用率 max_node_R1_util_rate 和 max_node_R2_util_rate
  std::vector<double> node_R1_util_rate_vector;
  std::vector<double> node_R2_util_rate_vector;
  for (const auto& value : m_node_attr_vector) {
    node_R1_util_rate_vector.push_back((value.D_R1 - value.D_R1_res) / value.D_R1);
    node_R2_util_rate_vector.push_back((value.D_R2 - value.D_R2_res) / value.D_R2);
  }
  auto max_nR1r = std::max_element(node_R1_util_rate_vector.begin(), node_R1_util_rate_vector.end());
  max_node_R1_util_rate = *max_nR1r;

  auto max_nR2r = std::max_element(node_R2_util_rate_vector.begin(), node_R2_util_rate_vector.end());
  max_node_R2_util_rate = *max_nR2r;
  // 使用偏度指标
  std::vector<double> data_std_R1;
  std::vector<double> data_std_R2;
  for(const auto& value : m_node_attr_vector) {
    data_std_R1.push_back(value.D_R1 - value.D_R1_res);
    data_std_R2.push_back(value.D_R2 - value.D_R2_res);
  }
  double r_std_R1 = standardDeviation(data_std_R1);
  double r_std_R2 = standardDeviation(data_std_R2);

  // 使用偏度指标 - 百分比利用率
  std::vector<double> data_std_R1_percent;
  std::vector<double> data_std_R2_percent;
  for(const auto& value : node_R1_util_rate_vector) {
    data_std_R1_percent.push_back(value);
  }
  for(const auto& value : node_R2_util_rate_vector) {
    data_std_R2_percent.push_back(value);
  }
  double r_std_R1_percent = standardDeviation(data_std_R1_percent);
  double r_std_R2_percent = standardDeviation(data_std_R2_percent);


//  // ### (10) ### 长期平均节点利用率
//// 最大化长期节点资源平均利用率(R1, R2) m_avg_node_util_R1_vector 和 m_avg_node_util_R2_vector的平均值
//// 本训练步之前节点平均利用率R1
//double avg_node_R1_util_rate_before_step_sum = std::accumulate(std::begin(m_avg_node_util_R1_vector), std::end(m_avg_node_util_R1_vector), 0.0);
//double avg_node_R1_util_rate_before_step_mean =  avg_node_R1_util_rate_before_step_sum / m_avg_node_util_R1_vector.size(); //R1节点长期利用率均值
//// 本训练步之前节点平均利用率R2
//double avg_node_R2_util_rate_before_step_sum = std::accumulate(std::begin(m_avg_node_util_R2_vector), std::end(m_avg_node_util_R2_vector), 0.0);
//double avg_node_R2_util_rate_before_step_mean =  avg_node_R2_util_rate_before_step_sum / m_avg_node_util_R2_vector.size(); //R2节点均值
//// 当前训练步节点平均利用率R1
//double avg_node_R1_util_rate_step_sum = std::accumulate(std::begin(node_R1_util_rate_vector), std::end(node_R1_util_rate_vector), 0.0);
//double avg_node_R1_util_rate_step_mean =  avg_node_R1_util_rate_step_sum / node_R1_util_rate_vector.size(); //R1节点长期利用率均值
//// 当前训练步节点平均利用率R2
//double avg_node_R2_util_rate_step_sum = std::accumulate(std::begin(node_R2_util_rate_vector), std::end(node_R2_util_rate_vector), 0.0);
//double avg_node_R2_util_rate_step_mean =  avg_node_R2_util_rate_step_sum / node_R2_util_rate_vector.size(); //R2节点均值
//if(m_avg_node_util_R1_vector.size() == 0) {
//  avg_node_R1_util_rate_before_step_mean = avg_node_R1_util_rate_step_mean;
//  avg_node_R2_util_rate_before_step_mean = avg_node_R2_util_rate_step_mean;
//}
//// 长期平均节点利用率收益计算，单步 - 长期平均值
//double r_avg_util_long = (avg_node_R1_util_rate_step_mean + avg_node_R2_util_rate_step_mean - avg_node_R1_util_rate_before_step_mean - avg_node_R2_util_rate_before_step_mean);
//
//
//m_avg_node_util_R1_vector.push_back(avg_node_R1_util_rate_step_mean);
//m_avg_node_util_R2_vector.push_back(avg_node_R2_util_rate_step_mean);

  // ** 重复指标
  double duplicationRate = calculateDuplicationRate(m_action_vector);
  std::cout << "====>>> duplicationRate = " << duplicationRate << std::endl;


  // ######## state_fixed ########
  double lr_r_std_R1;
  double lr_r_std_R2;
  if (m_log_state_fixed == true) {
    if (m_cnt_step <= m_state_buffer_size + 1) {
      lr_r_std_R1 = r_std_R1;
      lr_r_std_R2 = r_std_R2;
      m_l_rand_std_R1_map[m_cnt_VNR] = lr_r_std_R1;
      m_l_rand_std_R2_map[m_cnt_VNR] = lr_r_std_R2;
    } else {
      lr_r_std_R1 = m_l_rand_std_R1_map[m_cnt_VNR];
      lr_r_std_R2 = m_l_rand_std_R2_map[m_cnt_VNR];
    }
  }
  // ######## state_fixed ########

  // ### Reward ### r_Rev_to_Cost，r_load，r_acc

  double alpha = 1.0; // 0.5
  double beta = 0; // 0.5
  double gamma = 0.0; //0.7
  double theta = 0; //0.7

  std::cout << "====>>> value.D_R1 - value.D_R1_res  ";
  for(const auto& value : m_node_attr_vector) {
    std::cout << value.D_R1 - value.D_R1_res << " ";
  }
  std::cout << std::endl;
  std::cout << "====>>> value.D_R2 - value.D_R2_res  ";
  for(const auto& value : m_node_attr_vector) {
    std::cout << value.D_R2 - value.D_R2_res << " ";
  }
  std::cout << std::endl;

  // evel
  e_acc = log_acc;
  e_Cost_N = Cost_N;
  e_Rev_N = Rev_N;
  e_cnt_step = m_cnt_step;

  if (log_acc == 0) {
    NS_LOG_UNCOND( "==>> *************** get ************* log_acc = 0; reward = " << reward);
    e_Cost_N = 0;
    e_Rev_N = 0;
    reward = -1;
    return reward;
  }

  double r_Rev_to_Cost = Rev / Cost;
  double r_Rev_to_Cost_N = Rev_N / Cost_N;

  // double r_load = 0.0*(1 - max_link_util_rate) + 0.5 * (1 - r_std_) + 0.5 * 2 * (0.5 - r_std_R2);
  //double r_load = 0.0*(1 - max_link_util_rate) + (1-r_std_R1 / (1e-8+lr_r_std_R1)) + (1-r_std_R2 / (1e-8+lr_r_std_R2));
  // double r_load = 0;
  // 联动调整收益代价和节点利用率, 7/3开
  double r_load = 0.0*(1 - max_link_util_rate) + 0.7 * (1 - r_std_R1) + 0.3 * (1 - r_std_R2);

//  reward = alpha * r_Rev_to_Cost_N + beta * r_load;

  reward = alpha *  r_Rev_to_Cost_N + beta * r_load - 5.5*duplicationRate;// + gamma * m_acc_rate;// + theta * r_avg_util_long;

  // 联动调整收益代价和节点利用率, 7/3开
  //reward_scaling = (reward - 0) / (5.5 * 0.7 + 5 * 0.3);
//  reward_scaling = (reward - 0) / 0.5;
  //reward_scaling = (reward - 2) / 2;
  reward_scaling = (reward - 0) / 5.5;
  // reward_scaling = (reward - 0) / 1;

  std::cout <<"==>> *******==>> reward = " << reward << "; reward_scaling = " << reward_scaling
                << "; r_Rev_to_Cost = " << r_Rev_to_Cost_N << "; r_load = "<< r_load << std::endl;
  std::cout <<"==>> *******==>> Rev_N = " << Rev_N << "; Cost_N = " << Cost_N << std::endl;

  // std::cout <<"==>> *******==>> r_load = " << r_load << "; r_std_R1_percent = " << r_std_R1_percent
  //       << "; r_std_R2_percent = " << r_std_R2_percent << std::endl;

  // std::cout <<"==>> *******==>> r_std_R1 = " << r_std_R1 << "; r_std_R2 = " << r_std_R2 << std::endl;

//   NS_LOG_UNCOND("==>> *******==>> lr_r_std_R1 = " << lr_r_std_R1 << "; lr_r_std_R2 = " << lr_r_std_R2);

  // NS_LOG_UNCOND("==>> *******==>> 1 - max_link_util_rate = "<< 1 - max_link_util_rate <<"; 1 - max_node_R1_util_rate = "
  //               << 1 - max_node_R1_util_rate <<"; 1 / max_node_R2_util_rate = " << 1 - max_node_R2_util_rate);

  return reward_scaling;
}

/*
Define extra info. Optional
*/
std::string
MyGymEnv::GetExtraInfo()
{
  // e_cnt_step, e_acc, e_Rev_N, e_Cost_N
  std::string myInfo = "Info";
  myInfo += "|" + std::string("e_cnt_step=") + std::to_string(e_cnt_step);
  myInfo += "|" + std::string("e_acc=") + std::to_string(e_acc);
  myInfo += "|" + std::string("e_Rev_N=") + std::to_string(e_Rev_N);
  myInfo += "|" + std::string("e_Cost_N=") + std::to_string(e_Cost_N);
  myInfo += "|";
  NS_LOG_UNCOND("MyGetExtraInfo: " << myInfo);
  return myInfo;
}

void
MyGymEnv::ExeNetApp(uint32_t id_from_node_net, uint32_t id_to_node_net, double bandw_embed)
{
  // 数据传输（带宽占用）
  // 首先是sinkHelper，函数的第一个参数是协议，第二个是IP地址
  Ptr<Node> serverNode = m_nodes.Get(id_to_node_net);
  Address localAddress = InetSocketAddress(Ipv4Address::GetAny(), m_port_onoff);

  PacketSinkHelper sinkHelper = PacketSinkHelper ("ns3::UdpSocketFactory", localAddress);
  ApplicationContainer sinkApp = sinkHelper.Install (serverNode);
  sinkApp.Start (Seconds (m_curTime + 0.0));
  sinkApp.Stop (Seconds (m_curTime + m_life_cycle));

  //获取 sink_address_ipv4
  std::vector<std::string> v_node_pair_temper = m_nodeId_to_DevStr_map[id_to_node_net];
  std::string id_ip_interface = v_node_pair_temper[0];//！！这里仅从v_node_pair_temper选了第一个
  Ipv4Address sink_address_ipv4;
  // 将字符串中"_"前的字符取出并转为uint32_t类型，例如 "12_6" => 12
  size_t underscore_pos = id_ip_interface.find_first_of('_');
  std::string first_part = id_ip_interface.substr(0, underscore_pos);
  if ((uint32_t)std::stoi(first_part) == id_to_node_net) {
    sink_address_ipv4 = m_ip_interface_cont_map[id_ip_interface].GetAddress (0);
  } else {
    sink_address_ipv4 = m_ip_interface_cont_map[id_ip_interface].GetAddress (1);
  }

  //下面就是为各个节点创建应用程序，使用的是OnOffApplication
  Ptr<Node> clientNode = m_nodes.Get(id_from_node_net);
  Address sinkAddress = InetSocketAddress (sink_address_ipv4, m_port_onoff);

  // OnOff应用
  OnOffHelper clientHelper = OnOffHelper ("ns3::UdpSocketFactory", sinkAddress);
  clientHelper.SetConstantRate (DataRate (std::to_string(bandw_embed) + std::string("bps")));// ！！单位 bps
  clientHelper.SetAttribute ("PacketSize", UintegerValue (m_setPacketSize));//80//800// ！！单位 byte，根据m_interval和m_life_cycle的量级来调整
  clientHelper.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
  clientHelper.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
  ApplicationContainer clientApp = clientHelper.Install (clientNode);
  clientApp.Start (Seconds (m_curTime + 0.0));
  clientApp.Stop (Seconds (m_curTime + m_life_cycle));

  NS_LOG_UNCOND("sink_address_ipv4= " << sink_address_ipv4);
  NS_LOG_UNCOND("serverNode.GetId() = " << serverNode->GetId()<< "; clientNode.GetId() = " << clientNode->GetId()
  <<"; m_curTime = "<<m_curTime<<"; m_life_cycle = "<<m_life_cycle<<"; id_ip_interface = "<<id_ip_interface);
}


/*
Execute received actions
*/
bool
MyGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  // Ptr<OpenGymDictContainer> dict = DynamicCast<OpenGymDictContainer>(action);
  // Ptr<OpenGymBoxContainer<uint32_t> > box = DynamicCast<OpenGymBoxContainer<uint32_t> >(dict->Get("myActionVector"));
  // Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(dict->Get("myActionValue"));

  // NS_LOG_UNCOND ("MyExecuteActions: " << action);
  // NS_LOG_UNCOND ("---" << box);
  // NS_LOG_UNCOND ("---" << discrete);

  NS_LOG_UNCOND( "==>> ************** ExecuteActions begin ************ m_cnt_step = " << m_cnt_step <<"; m_cnt_VNR = "<< m_cnt_VNR << "; m_n_nodes_VNR = " <<m_n_nodes_VNR);
  //######################## Action 执行 ##############################
  // action: 一个VNR的所有虚拟节点的放置结果，m_n_max_VNF为一个VNR所能包含VNF的最大数量
  // 因此，action的维数固定为 m_n_max_VNF，并根据状态（state）做mask
  Ptr<OpenGymBoxContainer<uint32_t> > box_action = DynamicCast<OpenGymBoxContainer<uint32_t> >(action);
  m_action_vector = box_action->GetData();
  m_action_vector.resize(m_n_nodes_VNR); // 裁减去action_vector后面无效的部分

  std::cout << " action = ";
  for (const auto& value : m_action_vector) {
      std::cout << value << " ";
  }
  std::cout << std::endl;


  // if (m_port_onoff < 49000) {
  //   m_port_onoff = m_port_onoff + 1;
  // } else {
  //   m_port_onoff = 9001;
  // }

  // ######## state_fixed ########
  if (m_log_state_fixed == true) {
    if (m_cnt_step <= m_state_buffer_size) {
      m_node_attr_vector_sfmap[m_cnt_VNR] = m_node_attr_vector;
      m_node_cache_map_sfmap[m_cnt_VNR] = m_node_cache_map;
    } else {
      m_node_attr_vector = m_node_attr_vector_sfmap[m_cnt_VNR];
      m_node_cache_map = m_node_cache_map_sfmap[m_cnt_VNR];
    }
  }
  // ######## state_fixed ########


  std::vector<uint32_t> id_node_VNR_vector;
  for (const auto& pair : m_graphVNR.adjList) {
    id_node_VNR_vector.push_back(pair.first);
  }
  uint32_t id_node_VNR;
  uint32_t id_node_net;
  std::map<uint32_t, uint32_t> nodeId_VNR_to_net_map;
  AttrNodeVNR node_attr_VNR;
  std::vector<double> na_d_R1_vector (m_n_nodes);
  std::vector<double> na_d_R2_vector (m_n_nodes);
  // #### VNE，节点放置（嵌入）
  for (uint32_t i = 0; i < m_n_nodes_VNR; ++i) {

    id_node_VNR = id_node_VNR_vector[i]; //VNR中当前节点id
    id_node_net = m_action_vector[i]; //将VNR中当前节点放置（嵌入）至物理网络的节点id
    nodeId_VNR_to_net_map[id_node_VNR] = id_node_net;

    // 判断节点 id_node_net 是否部署了所请求的VNF
    node_attr_VNR = m_graphVNR.getNodeAttribute(id_node_VNR);
    // uint32_t req_type_VNF = node_attr_VNR.type_VNF; // VNF类型用（0，1，2，3，4）分别表示（”VNF_0“、”VNF_1“、”VNF_2“、”VNF_3、”VNF_4“）
    // std::vector<uint32_t> VNF_d_vector = m_node_VNF_d_vector[id_node_net];
    if (m_node_VNF_d_vector[id_node_net].at(node_attr_VNR.type_VNF) != 1) {
      std::cout << "==>> *************** get action .. m_log_typeVNF = false *************** " << std::endl;
      m_log_typeVNF = false; //false表示该net节点没有部署VNF
      break;
    } else {
      m_log_typeVNF = true;
    }

    //  NS_LOG_UNCOND( "==>> *************** get  ***************id_node_net = "<<id_node_net<<"; i = "<<i);
    //  NS_LOG_UNCOND( "==>> *************** m_node_attr_vector[id_node_net].D_R1_res *************** "<<m_node_attr_vector[id_node_net].D_R1_res);
    //  NS_LOG_UNCOND( "==>> *************** node_attr_VNR.d_R1 *************** "<<node_attr_VNR.d_R1);
    //  NS_LOG_UNCOND( "==>> *************** m_node_attr_vector[id_node_net].D_R2_res *************** "<<m_node_attr_vector[id_node_net].D_R2_res);
    //  NS_LOG_UNCOND( "==>> *************** node_attr_VNR.d_R2 *************** "<<node_attr_VNR.d_R2);
    // 判断剩余节点资源量是否满足所请求的资源量
    na_d_R1_vector[id_node_net] = na_d_R1_vector[id_node_net] + node_attr_VNR.d_R1; // 当同一个VNR中不同虚拟节点映射到同一个物理节点时，资源消耗累加
    na_d_R2_vector[id_node_net] = na_d_R2_vector[id_node_net] + node_attr_VNR.d_R2;
    if (m_node_attr_vector[id_node_net].D_R1_res - na_d_R1_vector[id_node_net] < 0
        || m_node_attr_vector[id_node_net].D_R2_res - na_d_R2_vector[id_node_net] < 0 ) {
      m_log_nodeRes = false; //资源需求超出net节点剩余资源
      std::cout << "==>> *************** get action .. m_log_nodeRes = false *************** " << std::endl;
      break;
    } else {
      m_log_nodeRes = true;
    }
  }

  // #### 更新net节点剩余资源和m_node_cache_map条目
  if (m_log_typeVNF && m_log_nodeRes) {
    for (uint32_t i = 0; i < m_n_nodes_VNR - 1; ++i) {

      id_node_VNR = id_node_VNR_vector[i]; //VNR中当前节点id
      id_node_net = m_action_vector[i]; //将VNR中当前节点放置（嵌入）至物理网络的节点id
      node_attr_VNR = m_graphVNR.getNodeAttribute(id_node_VNR);

      // ##### 更新net节点剩余资源
      // NodeAttr net_node_attr = m_node_attr_vector[id_node_net];
      m_node_attr_vector[id_node_net].D_R1_res = m_node_attr_vector[id_node_net].D_R1_res - node_attr_VNR.d_R1;
      m_node_attr_vector[id_node_net].D_R2_res = m_node_attr_vector[id_node_net].D_R2_res - node_attr_VNR.d_R2;

      // 新增 m_node_cache_map 条目
      // NodeStepCache node_step_cache(m_cnt_step, m_life_cycle, node_attr_VNR.d_R1, m_life_cycle, node_attr_VNR.d_R2);
      // m_node_cache_map[id_node_net][m_cnt_step] = node_step_cache;
      //注意如果一个VNR的不同虚拟节点对应同一个物理节点时，occupy_D_R1和occupy_D_R2累加，但res_time_R1和res_time_R2仍认为同为m_life_cycle（这里同一个VNR的虚拟节点放置后的执行时间认为是相同的）
      if (m_node_cache_map[id_node_net].find(m_cnt_step) == m_node_cache_map[id_node_net].end()) {
        m_node_cache_map[id_node_net][m_cnt_step] = {m_cnt_step, m_life_cycle, node_attr_VNR.d_R1, m_life_cycle, node_attr_VNR.d_R2};
      } else {
        m_node_cache_map[id_node_net][m_cnt_step].res_time_R1 = m_life_cycle; // 这里的res_time_R1认为该VNR所有放置在同一个物理节点VNF的服务时间都是相同的
        m_node_cache_map[id_node_net][m_cnt_step].occupy_D_R1 = m_node_cache_map[id_node_net][m_cnt_step].occupy_D_R1 + node_attr_VNR.d_R1;
        m_node_cache_map[id_node_net][m_cnt_step].res_time_R2 = m_life_cycle; // 这里的res_time_R2认为该VNR所有放置在同一个物理节点VNF的服务时间都是相同的
        m_node_cache_map[id_node_net][m_cnt_step].occupy_D_R2 = m_node_cache_map[id_node_net][m_cnt_step].occupy_D_R2 + node_attr_VNR.d_R2;
      }
      // NS_LOG_UNCOND( "==>> ***** m_node_cache_map[id_node_net][m_cnt_step].occupy_D_R1 = "<<m_node_cache_map[id_node_net][m_cnt_step].occupy_D_R1
      // << "; m_node_cache_map[id_node_net][m_cnt_step].occupy_D_R2 = "<<m_node_cache_map[id_node_net][m_cnt_step].occupy_D_R2);
    }
  }

  // 使用 m_node_cache_map 释放节点资源
  double res_t_1;
  double res_t_2;
  // bool log_R1_release;
  // bool log_R2_release;
  std::vector<uint32_t> key_erase_vector;
  // NS_LOG_UNCOND("m_n_nodes_VNR = "<<m_n_nodes_VNR<<"; ==>> *************** get 1 *************** m_node_cache_map.size = "<< m_node_cache_map.size());
  for (auto& pair_n_map : m_node_cache_map) {// 注意这里去掉了const
    // NS_LOG_UNCOND( "==>> *************** get Node *************** pair_n_map.second.size = "<<pair_n_map.second.size() << "; pair_n_map.first = " <<pair_n_map.first);

    for (auto& pair_n_step : pair_n_map.second) {
      // NS_LOG_UNCOND( "==>> *************** get 2 *************** pair_n_step.first = "<< pair_n_step.first
      //               << "; pair_n_map.first = " <<pair_n_map.first<< "; pair_n_step.second.res_time_R1 = " << pair_n_step.second.res_time_R1
      //               <<"; pair_n_step.second.res_time_R2 = " << pair_n_step.second.res_time_R2);
      // log_R1_release = false;
      // log_R2_release = false;
      res_t_1 = pair_n_step.second.res_time_R1 - m_interval;//R1剩余占用时间
      res_t_2 = pair_n_step.second.res_time_R2 - m_interval;//R2剩余占用时间

      if (res_t_1 <= 0 && pair_n_step.second.res_time_R1 > 0) {
        m_node_attr_vector[pair_n_map.first].D_R1_res = m_node_attr_vector[pair_n_map.first].D_R1_res
                                                          + pair_n_step.second.occupy_D_R1;//释放占用资源
        // log_R1_release = true;
      // NS_LOG_UNCOND( "==>> *************** get 3 *************** m_node_attr_vector[pair_n_map.first].D_R1_res = "
      // <<m_node_attr_vector[pair_n_map.first].D_R1_res<<"; pair_n_step.second.occupy_D_R1 = "<<pair_n_step.second.occupy_D_R1);
      }

      if (res_t_2 <= 0 && pair_n_step.second.res_time_R2 > 0) {
        m_node_attr_vector[pair_n_map.first].D_R2_res = m_node_attr_vector[pair_n_map.first].D_R2_res
                                                          + pair_n_step.second.occupy_D_R2;//释放占用资源
        // log_R2_release = true;
      // NS_LOG_UNCOND( "==>> *************** get 4 *************** m_node_attr_vector[pair_n_map.first].D_R2_res = "
      // <<m_node_attr_vector[pair_n_map.first].D_R2_res<<"; pair_n_step.second.occupy_D_R2 = "<<pair_n_step.second.occupy_D_R2);
      }

      pair_n_step.second.res_time_R1 = res_t_1;//更新剩余占用时间
      pair_n_step.second.res_time_R2 = res_t_2;//更新剩余占用时间
      // NS_LOG_UNCOND("res_t_1 = "<<res_t_1<<"; res_t_2 = "<<res_t_2
      //               << "==>> *************** get 5 *************** ");

      // if (res_t_1 <= 0 && res_t_2 <= 0 && pair_n_map.second.size() > 1 && log_R1_release != true && log_R2_release != true) {
      // if (res_t_1 <= 0 && res_t_2 <= 0 && pair_n_map.second.size() > 1) {
      if (res_t_1 <= 0 && res_t_2 <= 0) {
        key_erase_vector.push_back(pair_n_step.first);//删除此条目
        // pair_n_map.second.erase(pair_n_step.first);//删除此条目
        // NS_LOG_UNCOND(" ==>> *************** get erase *************** pair_n_step.first = "<<pair_n_step.first
        //         << "; pair_n_map.first = " <<pair_n_map.first);
      }
      // NS_LOG_UNCOND("; m_node_attr_vector[pair_n_map.first].D_R1_res = "<<m_node_attr_vector[pair_n_map.first].D_R1_res
      //               <<"; m_node_attr_vector[pair_n_map.first].D_R2_res = "<<m_node_attr_vector[pair_n_map.first].D_R2_res);
    }
    for (const auto& value : key_erase_vector) {
      pair_n_map.second.erase(value);
      // NS_LOG_UNCOND(" ==>> *************** ex erase *************** value = " << value);
    }
    key_erase_vector.clear();
    // NS_LOG_UNCOND( "==>> *************** get 6 *************** ");
  }
  // NS_LOG_UNCOND( "==>> *************** get end *************** ");





  // ## 声明网络发包部分的代码
  Time CurTime;
  Ptr<Node> serverNode;
  Address localAddress;
  PacketSinkHelper sinkHelper ("ns3::UdpSocketFactory", localAddress);
  ApplicationContainer sinkApp;
  std::vector<std::string> v_node_pair_temper;
  std::string id_ip_interface;
  Ipv4Address sink_address_ipv4;
  Ptr<Node> clientNode;
  Address sinkAddress;
  OnOffHelper clientHelper ("ns3::UdpSocketFactory", sinkAddress);
  ApplicationContainer clientApp;

  size_t underscore_pos;
  std::string first_part;

  // #### VNE，链路放置（嵌入）
  double bandw_embed; // VNR中，当前link嵌入的带宽需求
  double bandw_embed_UDP; // UDP下，为了得到实际的Tx网口发包速率而进行了调整，UDP为每个包增加了30byte
  uint32_t id_from_node_VNR; // VNR, from_node
  uint32_t id_to_node_VNR; // VNR, to_node
  uint32_t id_from_node_net; // net, from_node
  uint32_t id_to_node_net; // net, to_node
  std::map<uint32_t, double> to_id_attr_map;
  for (const auto& pair : m_graphVNR.adjList) {
    for (const auto& value : pair.second.outEdges) {
      to_id_attr_map[value.target] = value.edge_attribute.b_e;
    }
    for (const auto& pair_attr : to_id_attr_map) {

      bandw_embed = pair_attr.second; // VNR的link嵌入带宽需求
      bandw_embed_UDP = bandw_embed * (static_cast<double>(m_setPacketSize) / (static_cast<double>(m_setPacketSize) + 30));

      id_from_node_VNR = pair.first;
      id_to_node_VNR = pair_attr.first;
      id_from_node_net = nodeId_VNR_to_net_map[id_from_node_VNR];
      id_to_node_net = nodeId_VNR_to_net_map[id_to_node_VNR];

      // NS_LOG_UNCOND( "==>> ******** get ******** bandw_embed = "<<bandw_embed <<"; bandw_embed_UDP = "<<bandw_embed_UDP);
      // NS_LOG_UNCOND( "id_from_node_VNR = "<< id_from_node_VNR <<"; id_to_node_VNR = "<< id_to_node_VNR
      //                 << "; id_from_node_net = " << id_from_node_net << "; id_to_node_net = " << id_to_node_net);

      if (id_from_node_net == id_to_node_net) {
        continue; // 连续嵌入到同一个节点，则不用发送数据流
      }

      CurTime = Now ();//得到当前时间（Time 类型）
      m_curTime = CurTime.GetSeconds();//得到当前时间(double 类型）

      // MyGymEnv::ExeNetApp(id_from_node_net, id_to_node_net, bandw_embed);

      // 数据传输（带宽占用）
      // 首先是sinkHelper，函数的第一个参数是协议，第二个是IP地址
      serverNode = m_nodes.Get(id_to_node_net);
      localAddress = InetSocketAddress(Ipv4Address::GetAny(), m_port_onoff);

      sinkHelper = PacketSinkHelper ("ns3::UdpSocketFactory", localAddress);
      sinkApp = sinkHelper.Install (serverNode);
      sinkApp.Start (Seconds (0.0));
      sinkApp.Stop (Seconds (m_life_cycle));

      //获取 sink_address_ipv4
      v_node_pair_temper = m_nodeId_to_DevStr_map[id_to_node_net];
      id_ip_interface = v_node_pair_temper[0];//！！这里仅从v_node_pair_temper选了第一个
      // Ipv4Address sink_address_ipv4;
      // 将字符串中"_"前的字符取出并转为uint32_t类型，例如 "12_6" => 12
      underscore_pos = id_ip_interface.find_first_of('_');
      first_part = id_ip_interface.substr(0, underscore_pos);
      if ((uint32_t)std::stoi(first_part) == id_to_node_net) {
        sink_address_ipv4 = m_ip_interface_cont_map[id_ip_interface].GetAddress (0);
      } else {
        sink_address_ipv4 = m_ip_interface_cont_map[id_ip_interface].GetAddress (1);
      }

      //下面就是为各个节点创建应用程序，使用的是OnOffApplication
      clientNode = m_nodes.Get(id_from_node_net);
      sinkAddress = InetSocketAddress (sink_address_ipv4, m_port_onoff);

      // OnOff应用
      clientHelper = OnOffHelper ("ns3::UdpSocketFactory", sinkAddress);
      clientHelper.SetConstantRate (DataRate (std::to_string(bandw_embed_UDP) + std::string("bps")));// ！！单位 bps
      clientHelper.SetAttribute ("PacketSize", UintegerValue (m_setPacketSize));//80//800// ！！单位 byte，根据m_interval和m_life_cycle的量级来调整
      clientHelper.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
      clientHelper.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
      clientApp = clientHelper.Install (clientNode);
      clientApp.Start (Seconds (0.0));
      clientApp.Stop (Seconds (m_life_cycle));

      // NS_LOG_UNCOND("serverNode = " << serverNode->GetId()<< "; clientNode = " << clientNode->GetId()
      //               <<"; m_curTime = " << m_curTime << "; m_life_cycle = " << m_life_cycle);
      // NS_LOG_UNCOND("id_ip_interface = " << id_ip_interface <<"; sink_address_ipv4= " << sink_address_ipv4 << "; m_port_onoff = " << m_port_onoff);

      // 端口号更新
      if (m_port_onoff < 49000) {
        m_port_onoff = m_port_onoff + 1;
      } else {
        m_port_onoff = 9000;
      }

    }
    to_id_attr_map.clear();
  }
  // #########################################
  NS_LOG_UNCOND( "==>> ******************* ExecuteActions end ***************** ");

  return true;
}

} // ns3 namespace
