/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Piotr Gawlowicz
/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Piotr Gawlowicz
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
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 *
 */

#include "utlis.h"
#include "mygym.h"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <cassert>
#include <typeinfo>
#include <vector>
#include <map>
#include <tuple>
#include <unordered_map>

#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
// #include "ns3/csma-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
// #include "ns3/flow-monitor-module.h"

// #include "tutorial-app.h"


#include "ns3/opengym-module.h"
// #include "mygym.h"

/* Node topology
    n0 ----- n1 ----- n2 ----- n3
    |        |        |        |
    |        |        |        |
    n4 ----- n5 ----- n6 ----- n7
-*/

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("OpenGym");

//##################################### main{}
int
main (int argc, char *argv[])
{
  // Parameters of the scenario
  uint32_t simSeed = 1;
  double simulationTime = 1; //seconds
  double envStepTime = 0.1; //seconds, ns3gym env step time interval
  uint32_t openGymPort = 5555;
  uint32_t testArg = 0;

  uint32_t stepTotalNumber = 512 * 2000; //！！需要更改

  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
  // optional parameters
  cmd.AddValue ("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
  cmd.AddValue ("stepTime", "Gym Env step time in seconds. Default: 0.1s", envStepTime);
  cmd.AddValue ("testArg", "Extra simulation argument. Default: 0", testArg);
  cmd.Parse (argc, argv);

  NS_LOG_UNCOND("Ns3Env parameters:");
  NS_LOG_UNCOND("--simulationTime: " << simulationTime);
  NS_LOG_UNCOND("--openGymPort: " << openGymPort);
  NS_LOG_UNCOND("--envStepTime: " << envStepTime);
  NS_LOG_UNCOND("--seed: " << simSeed);
  NS_LOG_UNCOND("--testArg: " << testArg);

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);

//****// VNR
  // std::string vnr_path = "/home/ni4555/workspace/ns-3-allinone/ns-3.39/contrib/opengym/examples/opengym-2/dataset/generated_files/batch_task_cleaned.csv";
  std::cout << "------------------------------- 开始数据集处理 ----------------------------------------------" << std::endl;
  std::cout << "------------------------------- 开始节点读取 ----------------------------------------------" << std::endl;
  std::string vnr_path = "dataset/generated_files/batch_task_cleaned.csv";
  DataPreprocess dataPreprocess;
  std::vector<std::vector<std::string>> raw_data_vector = dataPreprocess.autoRead(vnr_path);
  //std::vector<std::vector<std::string>> raw_data_vector = dataPreprocess.readCsv(vnr_path);
  // std::unique_ptr<std::vector<std::vector<std::string>>>
  // raw_data_ptr = std::make_unique<std::vector<std::vector<std::string>>>(std::move(raw_data_vector));
  dataPreprocess.showContent();

  std::vector<std::string> row = raw_data_vector[0];
  for(const auto& col : row) {
    std::cout << col << " ";
  }
  std::cout << std::endl;

  std::cout << "The dataPreprocess has been generated"<< std::endl;
  std::cout << "raw_data_vector size = "<<raw_data_vector.size()<< std::endl;
  std::cout << "raw_data_vector size = "<<row.size()<< std::endl;

  std::cout << "------------------------------- 结束节点信息读取 ----------------------------------------------" << std::endl;

  std::cout << "------------------------------- 开始拓扑信息读取 ----------------------------------------------" << std::endl;
  DataPreprocess dataPreprocess_topo;
  std::string vnr_path_topology = "dataset/generated_files/generated_topology.csv";
  std::vector<std::vector<std::string>> raw_topology_vector = dataPreprocess_topo.autoRead(vnr_path_topology);
  //std::vector<std::vector<std::string>> raw_topology_vector = dataPreprocess_topo.readCsv(vnr_path_topology);
  dataPreprocess_topo.showContent();

  // 装配GraphVNR请求列表
  std::vector<GraphVNR> graphVNR_vector = dataPreprocess.dataToGraphVNR(raw_data_vector, raw_topology_vector, 20, 5);

  std::cout << "------------------------------- 结束拓扑信息读取 ----------------------------------------------" << std::endl;
  std::cout << "数据集VNR请求数量总计: " << graphVNR_vector.size() << std::endl;
  std::cout << "------------------------------- 数据集处理结束 ----------------------------------------------" << std::endl;



  // std::vector<GraphVNR> graphVNR_vector;
  // GraphVNR graphVNR;

  // // 添加节点和设置属性
  // graphVNR.addNode(0, AttrNodeVNR(0.9, 0.8, 2));
  // graphVNR.addNode(1, AttrNodeVNR(0.8, 0.4, 0));
  // graphVNR.addNode(2, AttrNodeVNR(0.7, 0.5, 1));
  // graphVNR.addNode(3, AttrNodeVNR(0.5, 0.7, 4));

  // // 添加有向边和设置属性
  // graphVNR.addEdge(0, 1, AttrLinkVNR(0.08 * 100*1000)); // 100*1000bps => 100Kbps
  // graphVNR.addEdge(1, 2, AttrLinkVNR(0.04 * 100*1000));
  // graphVNR.addEdge(2, 3, AttrLinkVNR(0.12 * 100*1000));

  // GraphVNR graphVNR1;
  // // 添加节点和设置属性
  // graphVNR1.addNode(0, AttrNodeVNR(0.9, 0.8, 2));
  // graphVNR1.addNode(1, AttrNodeVNR(0.8, 0.4, 0));
  // graphVNR1.addNode(2, AttrNodeVNR(0.7, 0.5, 1));
  // graphVNR1.addNode(3, AttrNodeVNR(0.5, 0.7, 4));
  // graphVNR1.addNode(4, AttrNodeVNR(0.5, 0.7, 3));
  // // 添加有向边和设置属性
  // graphVNR1.addEdge(0, 1, AttrLinkVNR(0.08 * 100*1000)); // 100*1000bps => 100Kbps
  // graphVNR1.addEdge(1, 2, AttrLinkVNR(0.04 * 100*1000));
  // graphVNR1.addEdge(1, 3, AttrLinkVNR(0.12 * 100*1000));
  // graphVNR1.addEdge(2, 4, AttrLinkVNR(0.10 * 100*1000));

  // GraphVNR graphVNR2;
  // // 添加节点和设置属性
  // graphVNR2.addNode(0, AttrNodeVNR(0.9, 0.8, 4));
  // graphVNR2.addNode(1, AttrNodeVNR(0.8, 0.4, 2));
  // graphVNR2.addNode(2, AttrNodeVNR(0.7, 0.5, 1));
  // // 添加有向边和设置属性
  // graphVNR2.addEdge(0, 1, AttrLinkVNR(0.07 * 100*1000)); // 100*1000bps => 100Kbps
  // graphVNR2.addEdge(0, 2, AttrLinkVNR(0.06 * 100*1000));

  // GraphVNR graphVNR3;
  // // 添加节点和设置属性
  // graphVNR3.addNode(0, AttrNodeVNR(0.9, 0.8, 4));
  // graphVNR3.addNode(1, AttrNodeVNR(0.8, 0.4, 2));
  // graphVNR3.addNode(2, AttrNodeVNR(0.7, 0.5, 1));
  // // 添加有向边和设置属性
  // graphVNR3.addEdge(0, 1, AttrLinkVNR(0.07 * 100*1000)); // 100*1000bps => 100Kbps
  // graphVNR3.addEdge(0, 2, AttrLinkVNR(0.06 * 100*1000));

  // GraphVNR graphVNR4;
  // // 添加节点和设置属性
  // graphVNR4.addNode(0, AttrNodeVNR(0.9, 0.8, 4));
  // graphVNR4.addNode(1, AttrNodeVNR(0.8, 0.4, 2));
  // graphVNR4.addNode(2, AttrNodeVNR(0.7, 0.5, 1));
  // // 添加有向边和设置属性
  // graphVNR4.addEdge(0, 1, AttrLinkVNR(0.08 * 100*1000)); // 100*1000bps => 100Kbps
  // graphVNR4.addEdge(1, 2, AttrLinkVNR(0.05 * 100*1000));

  // graphVNR_vector.push_back(graphVNR);
  // graphVNR_vector.push_back(graphVNR1);
  // graphVNR_vector.push_back(graphVNR2);
  // graphVNR_vector.push_back(graphVNR3);
  // graphVNR_vector.push_back(graphVNR4);



  //****// NETWORK
  // Create node
  // std::vector<EdgeConfig> edges = {{8, 10,"_","_"},// 节点和链路数目 // 注意 各个edge需遵循sort排序的顺序编写，以防后续代码报错
  //   {0,1,"100Kbps","2ms"},{0,4,"100Kbps","2ms"},{1,2,"100Kbps","2ms"},{1,5,"100Kbps","2ms"},{2,3,"100Kbps","2ms"},
  //   {2,6,"100Kbps","2ms"},{3,7,"100Kbps","2ms"},{4,5,"100Kbps","2ms"},{5,6,"100Kbps","2ms"},{6,7,"100Kbps","2ms"}
  // };//{节点编号i,节点编号j,带宽总量，link时延}

//     //GEANT
 std::vector<EdgeConfig> edges = {{23, 37,"_","_"},// 节点和链路数目 // 注意 各个edge需遵循sort排序的顺序编写，以防后续代码报错
   {0,10,"10Kbps","2ms"},{0,11,"10Kbps","2ms"},{0,13,"10Kbps","2ms"},{1,2,"10Kbps","2ms"},{1,13,"10Kbps","2ms"},
   {1,16,"10Kbps","2ms"},{1,20,"10Kbps","2ms"},{2,12,"10Kbps","2ms"},{2,14,"10Kbps","2ms"},{3,13,"10Kbps","2ms"},
   {3,20,"10Kbps","2ms"},{4,8,"10Kbps","2ms"},{4,19,"10Kbps","2ms"},{5,13,"10Kbps","2ms"},{5,21,"10Kbps","2ms"},
   {6,13,"10Kbps","2ms"},{6,15,"10Kbps","2ms"},{7,9,"10Kbps","2ms"},{7,12,"10Kbps","2ms"},{7,15,"10Kbps","2ms"},
   {8,15,"10Kbps","2ms"},{9,10,"10Kbps","2ms"},{9,16,"10Kbps","2ms"},{9,18,"10Kbps","2ms"},{9,21,"10Kbps","2ms"},
   {10,15,"10Kbps","2ms"},{10,18,"10Kbps","2ms"},{10,20,"10Kbps","2ms"},{11,17,"10Kbps","2ms"},{12,13,"10Kbps","2ms"},
   {12,14,"10Kbps","2ms"},{12,20,"10Kbps","2ms"}, {12,21,"10Kbps","2ms"},{15,19,"10Kbps","2ms"},{17,20,"10Kbps","2ms"},
   {17,22,"10Kbps","2ms"},{19,22,"10Kbps","2ms"},
 };//{节点编号i,节点编号j,带宽总量，link时延}

//    //Abilene
//   std::vector<EdgeConfig> edges = {{12, 15,"_","_"},// 节点和链路数目 // 注意 各个edge需遵循sort排序的顺序编写，以防后续代码报错
//     {0,1,"10Kbps","2ms"},{1,4,"10Kbps","2ms"},{1,5,"10Kbps","2ms"},{1,11,"10Kbps","2ms"},{2,5,"10Kbps","2ms"},
//     {2,8,"10Kbps","2ms"},{3,6,"10Kbps","2ms"},{3,9,"10Kbps","2ms"},{3,10,"10Kbps","2ms"},{4,6,"10Kbps","2ms"},
//     {4,7,"10Kbps","2ms"},{5,6,"10Kbps","2ms"},{7,9,"10Kbps","2ms"},{8,11,"10Kbps","2ms"},{9,10,"10Kbps","2ms"}
//   };//{节点编号i,节点编号j,带宽总量，link时延}

//     // ChinaNet
//    std::vector<EdgeConfig> edges = {{42, 66,"_","_"},// 节点和链路数目 // 注意 各个edge需遵循sort排序的顺序编写，以防后续代码报错
//     {0,3,"10Kbps","2ms"},{0,16,"10Kbps","2ms"},{0,39,"10Kbps","2ms"},{1,18,"10Kbps","2ms"},{1,39,"10Kbps","2ms"},
//     {2,33,"10Kbps","2ms"},{4,8,"10Kbps","2ms"},{5,38,"10Kbps","2ms"},{6,18,"10Kbps","2ms"},{6,39,"10Kbps","2ms"},
//     {7,39,"10Kbps","2ms"},{8,9,"10Kbps","2ms"},{8,11,"10Kbps","2ms"},{8,16,"10Kbps","2ms"},{8,18,"10Kbps","2ms"},
//     {8,23,"10Kbps","2ms"},{8,24,"10Kbps","2ms"},{8,25,"10Kbps","2ms"},{8,26,"10Kbps","2ms"},{8,27,"10Kbps","2ms"},
//     {8,28,"10Kbps","2ms"},{8,31,"10Kbps","2ms"},{8,38,"10Kbps","2ms"},{8,39,"10Kbps","2ms"},{9,27,"10Kbps","2ms"},
//     {10,39,"10Kbps","2ms"},{12,28,"10Kbps","2ms"},{13,25,"10Kbps","2ms"},{14,16,"10Kbps","2ms"},{14,28,"10Kbps","2ms"},
//     {15,16,"10Kbps","2ms"},{15,28,"10Kbps","2ms"},{16,27,"10Kbps","2ms"},{16,28,"10Kbps","2ms"},{17,28,"10Kbps","2ms"},
//     {18,25,"10Kbps","2ms"},{18,27,"10Kbps","2ms"},{18,28,"10Kbps","2ms"},{18,32,"10Kbps","2ms"},{18,33,"10Kbps","2ms"},
//     {18,39,"10Kbps","2ms"},{18,40,"10Kbps","2ms"},{18,41,"10Kbps","2ms"},{19,39,"10Kbps","2ms"},{20,23,"10Kbps","2ms"},
//     {21,28,"10Kbps","2ms"},{22,25,"10Kbps","2ms"},{22,28,"10Kbps","2ms"},{23,28,"10Kbps","2ms"},{23,39,"10Kbps","2ms"},
//     {25,27,"10Kbps","2ms"},{25,39,"10Kbps","2ms"},{27,30,"10Kbps","2ms"},{27,39,"10Kbps","2ms"},{28,29,"10Kbps","2ms"},
//     {28,38,"10Kbps","2ms"},{28,39,"10Kbps","2ms"},{32,39,"10Kbps","2ms"},{33,39,"10Kbps","2ms"},{34,39,"10Kbps","2ms"},
//     {35,39,"10Kbps","2ms"},{36,39,"10Kbps","2ms"},{37,38,"10Kbps","2ms"},{38,39,"10Kbps","2ms"},{39,40,"10Kbps","2ms"},
//     {39,41,"10Kbps","2ms"}
//    };//{节点编号i,节点编号j,带宽总量，link时延}

//    // Xspedius
//    std::vector<EdgeConfig> edges = {{34, 49,"_","_"},// 节点和链路数目 // 注意 各个edge需遵循sort排序的顺序编写，以防后续代码报错
//     {0,1,"10Kbps","2ms"},{0,3,"10Kbps","2ms"},{0,10,"10Kbps","2ms"},{0,20,"10Kbps","2ms"},{0,22,"10Kbps","2ms"},
//     {0,23,"10Kbps","2ms"},{1,6,"10Kbps","2ms"},{2,3,"10Kbps","2ms"},{2,31,"10Kbps","2ms"},{3,27,"10Kbps","2ms"},
//     {4,7,"10Kbps","2ms"},{4,8,"10Kbps","2ms"},{5,18,"10Kbps","2ms"},{5,24,"10Kbps","2ms"},{5,25,"10Kbps","2ms"},
//     {6,15,"10Kbps","2ms"},{6,21,"10Kbps","2ms"},{7,15,"10Kbps","2ms"},{7,23,"10Kbps","2ms"},{8,18,"10Kbps","2ms"},
//     {8,33,"10Kbps","2ms"},{9,18,"10Kbps","2ms"},{10,11,"10Kbps","2ms"},{10,14,"10Kbps","2ms"},{11,12,"10Kbps","2ms"},
//     {12,23,"10Kbps","2ms"},{13,24,"10Kbps","2ms"},{13,32,"10Kbps","2ms"},{14,19,"10Kbps","2ms"},{14,23,"10Kbps","2ms"},
//     {16,17,"10Kbps","2ms"},{16,18,"10Kbps","2ms"},{16,23,"10Kbps","2ms"},{17,19,"10Kbps","2ms"},{17,23,"10Kbps","2ms"},
//     {20,21,"10Kbps","2ms"},{21,22,"10Kbps","2ms"},{23,31,"10Kbps","2ms"},{24,31,"10Kbps","2ms"},{24,33,"10Kbps","2ms"},
//     {25,31,"10Kbps","2ms"},{26,27,"10Kbps","2ms"},{26,29,"10Kbps","2ms"},{27,28,"10Kbps","2ms"},{28,30,"10Kbps","2ms"},
//     {29,30,"10Kbps","2ms"},{30,31,"10Kbps","2ms"},{30,32,"10Kbps","2ms"},{32,33,"10Kbps","2ms"}
//    };//{节点编号i,节点编号j,带宽总量，link时延}


 // GEANT
 std::vector<NodeAttr> node_attr_vector = {
 //  {10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},
 //  {10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},
 //  {10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},
 //  {10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},
 //  {10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1}

  //  {10,10,0.3,10,10,0.3},{10,10,0.8,10,10,0.8},{10,10,0.3,10,10,0.3},{10,10,0.4,10,10,0.4},{10,10,0.9,10,10,0.9},
  //  {10,10,0.2,10,10,0.2},{10,10,0.3,10,10,0.3},{10,10,0.8,10,10,0.8},{10,10,0.8,10,10,0.8},{10,10,0.3,10,10,0.3},
  //  {10,10,0.8,10,10,0.8},{10,10,0.2,10,10,0.3},{10,10,0.2,10,10,0.2},{10,10,0.3,10,10,0.3},{10,10,0.4,10,10,0.4},
  //  {10,10,0.9,10,10,0.9},{10,10,0.7,10,10,0.7},{10,10,0.4,10,10,0.4},{10,10,0.9,10,10,0.9},{10,10,0.8,10,10,0.7},
  //  {10,10,0.5,10,10,0.5},{10,10,0.1,10,10,0.1},{10,10,0.8,10,10,0.8}

   {5,5,0.3,5,5,0.3},{5,5,0.8,5,5,0.8},{5,5,0.3,5,5,0.3},{5,5,0.4,5,5,0.4},{5,5,0.9,5,5,0.9},
   {5,5,0.2,5,5,0.2},{5,5,0.3,5,5,0.3},{5,5,0.8,5,5,0.8},{5,5,0.8,5,5,0.8},{5,5,0.3,5,5,0.3},
   {5,5,0.8,5,5,0.8},{5,5,0.2,5,5,0.3},{5,5,0.2,5,5,0.2},{5,5,0.3,5,5,0.3},{5,5,0.4,5,5,0.4},
   {5,5,0.9,5,5,0.9},{5,5,0.7,5,5,0.7},{5,5,0.4,5,5,0.4},{5,5,0.9,5,5,0.9},{5,5,0.8,5,5,0.7},
   {5,5,0.5,5,5,0.5},{5,5,0.1,5,5,0.1},{5,5,0.8,5,5,0.8}//！！！

//   {6,6,0.3,6,6,0.3},{6,6,0.8,6,6,0.8},{6,6,0.3,6,6,0.3},{6,6,0.4,6,6,0.4},{6,6,0.9,6,6,0.9},
//   {6,6,0.2,6,6,0.2},{6,6,0.3,6,6,0.3},{6,6,0.8,6,6,0.8},{6,6,0.8,6,6,0.8},{6,6,0.3,6,6,0.3},
//   {6,6,0.8,6,6,0.8},{6,6,0.2,6,6,0.3},{6,6,0.2,6,6,0.2},{6,6,0.3,6,6,0.3},{6,6,0.4,6,6,0.4},
//   {6,6,0.9,6,6,0.9},{6,6,0.7,6,6,0.7},{6,6,0.4,6,6,0.4},{6,6,0.9,6,6,0.9},{6,6,0.8,6,6,0.7},
//   {6,6,0.5,6,6,0.5},{6,6,0.1,6,6,0.1},{6,6,0.8,6,6,0.8}

   // {5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},
   // {5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},
   // {5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},
   // {5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},
   // {5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1}

//  {5,3,0.3,3,2,0.3},{5,4,0.8,4,3,0.8},{5,3,0.3,3,2,0.3},{5,2,0.4,5,3,0.4},{5,5,0.9,5,5,0.9},
//  {3,2,0.2,3,2,0.2},{5,5,0.3,5,5,0.3},{5,3,0.8,5,2,0.8},{5,3,0.8,4,3,0.8},{5,3,0.3,5,3,0.3},
//  {6,5,0.8,5,5,0.8},{4,4,0.2,5,5,0.3},{5,5,0.2,4,4,0.2},{5,5,0.3,5,3,0.3},{5,4,0.4,5,4,0.4},
//  {6,6,0.9,6,6,0.9},{6,6,0.7,6,6,0.7},{5,5,0.4,5,5,0.4},{6,6,0.9,6,6,0.9},{5,5,0.8,5,5,0.7},
//  {5,4,0.5,3,3,0.5},{5,2,0.1,4,2,0.1},{4,4,0.8,4,2,0.8}
 };//{资源总量,剩余总量，资源的单位cost}

//   // Abilene
//   std::vector<NodeAttr> node_attr_vector = {
//   //  {10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},
//   //  {10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},{10,9,1,10,9,1},
//   //  {10,9,1,10,9,1},{10,9,1,10,9,1}
//    {10,10,0.7,10,10,0.7},{10,10,0.9,10,10,0.9},{10,10,0.3,10,10,0.3},{10,10,0.2,10,10,0.2},{10,10,0.9,10,10,0.9},
//    {10,10,0.2,10,10,0.2},{10,10,0.8,10,10,0.8},{10,10,0.2,10,10,0.2},{10,10,0.8,10,10,0.8},{10,10,0.1,10,10,0.1},
//    {10,10,1.0,10,10,1.0},{10,10,0.2,10,10,0.2}
//     // {5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},
//     // {5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},{5,4,1,5,4,1},
//     // {5,4,1,5,4,1},{5,4,1,5,4,1}
//   };//{资源总量,剩余总量，资源的单位cost}

//      // ChinaNet
//   std::vector<NodeAttr> node_attr_vector = {
//     {5,5,0.3,5,5,0.3},{5,5,0.8,5,5,0.8},{5,5,0.3,5,5,0.3},{5,5,0.4,5,5,0.4},{5,5,0.9,5,5,0.9},
//     {5,5,0.2,5,5,0.2},{5,5,0.3,5,5,0.3},{5,5,0.8,5,5,0.8},{5,5,0.8,5,5,0.8},{5,5,0.2,5,5,0.2},
//     {5,5,0.8,5,5,0.8},{5,5,0.2,5,5,0.2},{5,5,0.1,5,5,0.1},{5,5,0.2,5,5,0.2},{5,5,0.4,5,5,0.4},
//     {5,5,0.9,5,5,0.9},{5,5,0.7,5,5,0.7},{5,5,0.7,5,5,0.7},{5,5,0.9,5,5,0.9},{5,5,0.8,5,5,0.8},
//     {5,5,0.3,5,5,0.3},{5,5,0.8,5,5,0.8},{5,5,0.3,5,5,0.3},{5,5,0.1,5,5,0.1},{5,5,0.7,5,5,0.7},
//     {5,5,0.1,5,5,0.1},{5,5,0.3,5,5,0.3},{5,5,0.8,5,5,0.8},{5,5,0.8,5,5,0.8},{5,5,0.3,5,5,0.3},
//     {5,5,0.8,5,5,0.8},{5,5,0.2,5,5,0.2},{5,5,0.2,5,5,0.2},{5,5,0.3,5,5,0.3},{5,5,0.1,5,5,0.1},
//     {5,5,0.7,5,5,0.7},{5,5,0.7,5,5,0.7},{5,5,0.9,5,5,0.9},{5,5,0.7,5,5,0.7},{5,5,0.3,5,5,0.3},
//     {5,5,0.4,5,5,0.4},{5,5,0.2,5,5,0.2}
//   };//{资源总量,剩余总量，资源的单位cost}

//         // Xspedius
//   std::vector<NodeAttr> node_attr_vector = {
//     {5,5,0.3,5,5,0.3},{5,5,0.8,5,5,0.8},{5,5,0.3,5,5,0.3},{5,5,0.4,5,5,0.4},{5,5,0.9,5,5,0.9},
//     {5,5,0.2,5,5,0.2},{5,5,0.3,5,5,0.3},{5,5,0.8,5,5,0.8},{5,5,0.8,5,5,0.8},{5,5,0.2,5,5,0.2},
//     {5,5,0.8,5,5,0.8},{5,5,0.2,5,5,0.2},{5,5,0.1,5,5,0.1},{5,5,0.2,5,5,0.2},{5,5,0.4,5,5,0.4},
//     {5,5,0.9,5,5,0.9},{5,5,0.7,5,5,0.7},{5,5,0.7,5,5,0.7},{5,5,0.9,5,5,0.9},{5,5,0.8,5,5,0.8},
//     {5,5,0.3,5,5,0.3},{5,5,0.8,5,5,0.8},{5,5,0.3,5,5,0.3},{5,5,0.1,5,5,0.1},{5,5,0.7,5,5,0.7},
//     {5,5,0.1,5,5,0.1},{5,5,0.3,5,5,0.3},{5,5,0.8,5,5,0.8},{5,5,0.8,5,5,0.8},{5,5,0.3,5,5,0.3},
//     {5,5,0.8,5,5,0.8},{5,5,0.2,5,5,0.2},{5,5,0.9,5,5,0.9},{5,5,0.3,5,5,0.3}
//   };//{资源总量,剩余总量，资源的单位cost}




// GEANT
 std::vector<std::vector<uint32_t>> node_VNF_deployment_vector = {
   {0,1,1,1,1},{1,0,1,1,1},{1,1,1,1,0},{1,1,0,1,1},{1,1,1,0,1},
   {1,0,1,1,1},{0,1,1,1,1},{1,1,1,0,1},{0,1,1,1,1},{1,1,0,1,1},
   {1,1,0,1,1},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,0,1},{1,1,1,0,1},
   {1,0,1,1,1},{0,1,1,1,1},{1,1,1,0,1},{1,1,0,1,1},{1,1,0,1,1},
   {0,1,1,1,1},{1,0,1,1,0},{0,1,1,1,1}
  //  {1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},
  //  {1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},
  //  {1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},
  //  {1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},
  //  {1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}
 };//{0,1,0,0,0}表示该节点上部署了”VNF_1“；一共5种不同的VNF（”VNF_0“、”VNF_1“、”VNF_2“、”VNF_3、”VNF_4“）

//   // Abilene
//   std::vector<std::vector<uint32_t>> node_VNF_deployment_vector = {
// //    {0,1,1,1,1},{1,0,1,1,1},{1,1,1,1,0},{1,1,0,1,1},{1,1,1,0,1},
// //    {1,0,1,1,1},{0,1,1,1,1},{1,1,1,0,1},{0,1,1,1,1},{1,1,0,1,1},
// //    {1,1,0,1,1},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,0,1},{1,1,1,0,1},
//     {0,1,1,1,1},{1,0,1,1,1},{1,1,1,1,0},{1,1,0,1,1},{1,1,1,0,1},
//     {1,0,1,1,1},{0,1,1,1,1},{1,1,1,0,1},{1,1,1,1,0},{1,1,0,1,1},
//     {1,1,1,1,0},{1,1,1,0,1}
//   };//{0,1,0,0,0}表示该节点上部署了”VNF_1“；一共5种不同的VNF（”VNF_0“、”VNF_1“、”VNF_2“、”VNF_3、”VNF_4“）

//   // ChinaNet
//   std::vector<std::vector<uint32_t>> node_VNF_deployment_vector = {
//     {0,1,1,1,1},{1,0,1,1,1},{1,1,1,1,0},{1,1,0,1,1},{1,1,1,0,1},
//     {1,0,1,1,1},{0,1,1,1,1},{1,1,1,0,1},{0,1,1,1,1},{1,1,0,1,1},
//     {1,1,0,1,1},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,0,1},{1,1,1,0,1},
//     {1,0,1,1,1},{0,1,1,1,1},{1,1,1,0,1},{1,1,0,1,1},{1,1,0,1,1},
//     {0,1,1,1,1},{1,0,1,1,0},{0,1,1,1,1},{1,1,1,1,0},{1,1,1,0,1},
//     {0,1,1,1,1},{1,0,1,1,1},{1,1,1,1,0},{1,1,0,1,1},{1,1,1,0,1},
//     {1,0,1,1,1},{0,1,1,1,1},{1,1,1,0,1},{0,1,1,1,1},{1,1,0,1,1},
//     {1,1,0,1,1},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,0,1},{1,1,1,0,1},
//     {1,0,1,1,1},{1,1,0,1,1}
//   };//{0,1,0,0,0}表示该节点上部署了”VNF_1“；一共5种不同的VNF（”VNF_0“、”VNF_1“、”VNF_2“、”VNF_3、”VNF_4“）

//     // Xspedius
//     std::vector<std::vector<uint32_t>> node_VNF_deployment_vector = {
//       {0,1,1,1,1},{1,0,1,1,1},{1,1,1,1,0},{1,1,0,1,1},{1,1,1,0,1},
//       {1,0,1,1,1},{0,1,1,1,1},{1,1,1,0,1},{0,1,1,1,1},{1,1,0,1,1},
//       {1,1,0,1,1},{1,1,1,1,0},{1,1,1,1,0},{1,1,1,0,1},{1,1,1,0,1},
//       {1,0,1,1,1},{0,1,1,1,1},{1,1,1,0,1},{1,1,0,1,1},{1,1,0,1,1},
//       {0,1,1,1,1},{1,0,1,1,0},{0,1,1,1,1},{1,1,1,1,0},{1,1,1,0,1},
//       {0,1,1,1,1},{1,0,1,1,1},{1,1,1,1,0},{1,1,0,1,1},{1,1,1,0,1},
//       {1,0,1,1,1},{0,1,1,1,1},{1,1,1,0,1},{0,1,1,1,1}
//     };//{0,1,0,0,0}表示该节点上部署了”VNF_1“；一共5种不同的VNF（”VNF_0“、”VNF_1“、”VNF_2“、”VNF_3、”VNF_4“）




  uint32_t node_num = edges.at(0).node_i;
  edges.erase(edges.begin());

  // 创建节点
  NS_LOG_INFO ("Create nodes.");
  NodeContainer nodes;
  nodes.Create (node_num);

  PointToPointHelper pointToPoint;
  Ipv4AddressHelper address;

  InternetStackHelper stack;
  stack.Install (nodes);

  // 创建节点对
  // std::map<std::string, Ptr<NetConfig>> node_pair_map;
  // 动态设置变量名
  std::map<std::string, NodeContainer> node_container_map;
  std::map<std::string, NetDeviceContainer> device_container_map;
  std::map<std::string, Ipv4InterfaceContainer> ip_interface_container_map;

  std::map<std::string, LinkAttr> link_attr_map;
  std::vector<std::string> link_tmp_name_vector;

  uint32_t cnt = 1;
  uint32_t count_IP_assign = 0;
  std::string address_base_str;
  for(const auto& edge : edges) {
    // 设置变量名
    std::string tmp_name = std::to_string(edge.node_i) + std::string("_") + std::to_string(edge.node_j);
    NS_LOG_UNCOND("tmp_name = " << tmp_name);

    node_container_map[tmp_name] = NodeContainer (nodes.Get (edge.node_i), nodes.Get (edge.node_j));

    pointToPoint.SetDeviceAttribute ("DataRate", StringValue (edge.data_rate));
    pointToPoint.SetChannelAttribute ("Delay", StringValue (edge.delay));
    device_container_map[tmp_name] = pointToPoint.Install (node_container_map[tmp_name]);

    std::string tmp_ip_string = "10.1." + std::to_string(cnt++) + ".0";
    address.SetBase (Ipv4Address(tmp_ip_string.data()),"255.255.255.0","0.0.0.1");
    ip_interface_container_map[tmp_name] = address.Assign (device_container_map[tmp_name]);

    NS_LOG_UNCOND("print: address (0) = " << ip_interface_container_map[tmp_name].GetAddress (0));
    NS_LOG_UNCOND("print: address (1) = " << ip_interface_container_map[tmp_name].GetAddress (1));
    //node_pair_map[tmp_name] = CreateObject<NetConfig> (node_container_map[tmp_name],pointToPoint,address,edge.node_i,edge.node_j,edge.data_rate,edge.delay);
    //// Ptr<NetConfig> node_pair = CreateObject<NetConfig> (nodes,edge.node_i,edge.node_i,edge.data_rate,edge.delay);

    // 链路属性： 一个边包括 正向、反向 两条链路
    std::string dr_str = edge.data_rate;
    double dr_dou = stod( dr_str.erase(dr_str.length()-4,4) );//例如：“100Kpbs” 取出 100，注意data_rate的单位应该统一为 bps
    // LinkAttr linkAttr_1  = { .B = dr_dou * 1000 * 1000};
    LinkAttr linkAttr_1  = {dr_dou * 1000 , dr_dou * 1000, 1};
    link_attr_map[tmp_name] = linkAttr_1;

    std::string tmp_name_2 = std::to_string(edge.node_j) + std::string("_") + std::to_string(edge.node_i);
    // LinkAttr linkAttr_2  = { .B = dr_dou * 1000 * 1000 };
    LinkAttr linkAttr_2  = {dr_dou * 1000, dr_dou * 1000, 1};
    link_attr_map[tmp_name_2] = linkAttr_2;

    link_tmp_name_vector.push_back(tmp_name);
    link_tmp_name_vector.push_back(tmp_name_2);

    uint32_t num_interfaces = ip_interface_container_map[tmp_name].GetN();
    count_IP_assign += num_interfaces;

    NS_LOG_UNCOND("cnt = " << cnt);
  }

  // NetDeviceContainer dev_c = device_container_map["0_1"];
  // Ptr<PointToPointNetDevice> ptr_dev_0_1 = dev_c.Get(0);
  // ptr_dev_0_1->TraceConnectWithoutContext


  // ### 对 vector 进行字典序排序
  // std::sort(link_tmp_name_vector.begin(), link_tmp_name_vector.end());
  std::sort(link_tmp_name_vector.begin(), link_tmp_name_vector.end(), compareStringsByNumbers);
  // 初始化 map 来存储排序后的字符串及其对应的索引（作为值）
  std::map<std::string, uint32_t> link_str_to_idx_map;
  uint32_t idx_sort = 0;
  // 遍历排序后的 vector，并将每个字符串及其索引添加到 map 中
  for (const std::string& str_tmp : link_tmp_name_vector) {
      link_str_to_idx_map[str_tmp] = idx_sort;
      std::cout << "str_tmp \"" << str_tmp << "\" => idx_sort: " << idx_sort << std::endl;
      ++idx_sort;
  }
  // 打印结果
  for (const auto& pair : link_str_to_idx_map) {
      std::cout << "键 \"" << pair.first << "\" => 值: " << pair.second << std::endl;
  }

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // Enable ECMP by setting the attribute on the Ipv4RoutingHelper
  // Ipv4GlobalRoutingHelper::SetEcmp(true);

  // NS_LOG_UNCOND("----:------");
  // NS_LOG_INFO ("Create Applications.");
  // uint16_t port_onoff = 5000;   // Discard port (RFC 863)

  // //首先是sinkHelper，函数的第一个参数是协议，第二个是IP地址
  // Ptr<Node> serverNode = nodes.Get(7);
  // Address localAddress = InetSocketAddress(Ipv4Address::GetAny(), port_onoff);
  // // Address sinkLocalAddress = InetSocketAddress (node_pair_map["ptr_n2_n3"]->interfaces.GetAddress (1), port_onoff);
  // //std::cout << "ptr_n0_n1 = (0) " << node_pair_map["ptr_n0_n1"]->interfaces.GetAddress (0) << std::endl;

  // PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory", localAddress);
  // ApplicationContainer sinkApp = sinkHelper.Install (serverNode);
  // sinkApp.Start (Seconds (0.0));
  // sinkApp.Stop (Seconds (150.0));

  // //下面就是为各个节点创建应用程序，使用的是OnOffApplication
  // Ptr<Node> clientNode = nodes.Get(1);
  // Address sinkAddress = InetSocketAddress (ip_interface_container_map["6_7"].GetAddress (1), port_onoff);
  // //std::cout << "ptr_n0_n1 = (0) " << node_pair_map["ptr_n0_n1"]->interfaces.GetAddress (0) << std::endl;

  // // OnOff应用
  // OnOffHelper clientHelper ("ns3::TcpSocketFactory", sinkAddress);
  // clientHelper.SetConstantRate (DataRate ("1000bps"));
  // clientHelper.SetAttribute ("PacketSize", UintegerValue (125));
  // clientHelper.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
  // clientHelper.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
  // ApplicationContainer clientApp = clientHelper.Install (clientNode);
  // clientApp.Start (Seconds (1.0));
  // clientApp.Stop (Seconds (149.0));


  // // TcpSocket应用
  // // Ptr<Socket> ns3TcpSocket = Socket::CreateSocket(clientNode, TcpSocketFactory::GetTypeId());
  // // ns3TcpSocket->TraceConnectWithoutContext("CongestionWindow", MakeCallback(&CwndChange));

  // // Ptr<TutorialApp> app = CreateObject<TutorialApp>();//此句代码报错
  // // app->Setup(ns3TcpSocket, sinkAddress, 125, 1000, DataRate("1000bps"));//(Ptr<Socket> socket, Address address, uint32_t packetSize, uint32_t nPackets, uint32_t DataRate dataRate)
  // // clientNode->AddApplication(app);
  // // app->SetStartTime(Seconds(1.));
  // // app->SetStopTime(Seconds(100.));



  // // device_container_map["ptr_n2_n4"].Get(1)->TraceConnectWithoutContext("PhyRxDrop", MakeCallback(&RxDrop));

  // // std::ostringstream oss;

  // // Ptr<Node> appSource = NodeList::GetNode(backboneNodes);
  // // oss <<"/NodeList/" << nodes.Get(0)->GetId () << "/DeviceList/" << device_container_map["2_3"].Get(1)->GetNode()->GetId()
  // // << "/$ns3::PointToPointNetDevice/MacTx";

  // // oss << "/NodeList/"<< "*" << "/DeviceList/" << device_container_map["0_1"].Get(0)->GetAddress()
  // //     << "/$ns3::PointToPointNetDevice/MacTx";
  // // // oss <<"/NodeList/" << nodes.Get(1)->GetId () << "/$ns3::PointToPointNetDevice/MacTx";
  // // Config::Connect (oss.str (), MakeCallback (&TraceNetStats));

  // // // Simulator::Schedule (Seconds (m_monitor_interval_time), &TraceNetStats);



  //**// OpenGym Env

  // // Install FlowMonitor on all nodes
  // FlowMonitorHelper flowmon;
  // // std::map<std::string, FlowMonitorHelper> flowmon_map = {};
  // Ptr<FlowMonitor> monitor = flowmon.InstallAll();
  // monitor->CheckForLostPackets();
  // Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());

  Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface> (openGymPort);

  Ptr<MyGymEnv> myGymEnv = CreateObject<MyGymEnv> (nodes,
                                                  device_container_map,
                                                  ip_interface_container_map,
                                                  node_attr_vector,
                                                  node_VNF_deployment_vector,
                                                  link_attr_map,
                                                  link_str_to_idx_map,
                                                  graphVNR_vector,
                                                  // monitor,
                                                  // classifier,
                                                  simulationTime,
                                                  stepTotalNumber);

  myGymEnv->SetOpenGymInterface(openGymInterface);

  NS_LOG_UNCOND ("Simulation start");
  Simulator::Stop (Seconds (simulationTime));
  Simulator::Run ();
  NS_LOG_UNCOND ("Simulation stop");

  openGymInterface->NotifySimulationEnd();
  Simulator::Destroy ();

}
