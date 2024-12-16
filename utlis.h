#ifndef UTLIS_SIM_MYGYM_H
#define UTLIS_SIM_MYGYM_H

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <typeinfo>
#include <vector>
#include <tuple>
#include <map>
#include <unordered_map>
#include <memory> // 包含unique_ptr的定义

#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/csma-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"

// #include "tutorial-app.h"
//####################################其他功能函数
bool compareStringsByNumbers(const std::string& a, const std::string& b);
double skewness(const std::vector<double>& data);
double standardDeviation(const std::vector<double>& data);
double calculateDuplicationRate(const std::vector<uint32_t> &vec);

//####
namespace ns3 {

class NetConfig : public Object
{
  public:
    NetConfig (
    ns3::NodeContainer& node_container,
    ns3::PointToPointHelper& pointToPoint,
    ns3::Ipv4AddressHelper& address,
    uint32_t nodeID_i, uint32_t nodeID_j,
    std::string device_attr, std::string channel_attr
    );

    virtual ~NetConfig ();
    static TypeId GetTypeId (void);
    std::string addToIPv4Address(const std::string& ipv4Address, uint32_t increment);

    //ns3::NodeContainer node_container;
    // ns3::PointToPointHelper pointToPoint;
    ns3::NetDeviceContainer devices;
    // ns3::InternetStackHelper stack;
    // ns3::Ipv4AddressHelper address;
    ns3::Ipv4InterfaceContainer interfaces;

    static uint32_t count_instant;
    static uint32_t count_IP_assign;
    static std::string address_base_str;
};

//##################################### 任务VNR，以一个有向无环图的形式存储一个任务VNR的信息

struct AttrNodeVNR {

  double d_R1; //VNR中该虚拟节点的对于资源R1的需求总量
  double d_R2; //VNR中该虚拟节点的对于资源R2的需求总量
  uint32_t type_VNF; // VNF类型用（0，1，2，3，4）分别表示（”VNF_0“、”VNF_1“、”VNF_2“、”VNF_3、”VNF_4“）

  AttrNodeVNR() = default;
  AttrNodeVNR(double dr1, double dr2, uint32_t t_vnf) : d_R1(dr1), d_R2(dr2), type_VNF(t_vnf) {}
};

struct AttrLinkVNR {
  double b_e; //VNR 对于该网络链路的带宽需求总量，单位 bps

  AttrLinkVNR() = default;
  AttrLinkVNR(double be) : b_e(be) {}
};

// 边结构体，包含边的属性和目标节点
struct EdgeVNR {
  uint32_t target; // 目标节点的ID
  AttrLinkVNR edge_attribute; // 边的权重 d_R1

   // 默认构造函数
  EdgeVNR() = default;
  // 可以添加更多边相关的属性
  EdgeVNR(uint32_t t, AttrLinkVNR edge_attr) : target(t), edge_attribute(edge_attr) {}
};

// 节点结构体，包含节点属性和出边列表
struct NodeVNR {
  uint32_t id; // 节点的唯一标识符
  AttrNodeVNR node_attribute; // 节点的属性（请求的VNF相关信息）
  std::vector<EdgeVNR> outEdges; // 出边列表

  // 默认构造函数
  NodeVNR() = default;
  // 可以添加更多节点相关的属性和方法
  NodeVNR(uint32_t idx, AttrNodeVNR attr) : id(idx), node_attribute(attr) {}

  // 添加出边的方法
  void addOutEdge(uint32_t target, AttrLinkVNR edge_attribute) {
    outEdges.push_back(EdgeVNR(target, edge_attribute));
  }
};

// 图类，使用邻接表存储一个有向无环图
class GraphVNR : public Object
{
  // private:
  //   std::map<uint32_t, NodeVNR> adjList; // 邻接表

  public:
    GraphVNR ();
    virtual ~GraphVNR ();
    static TypeId GetTypeId (void);

    std::map<uint32_t, NodeVNR> adjList; // 邻接表

    // 添加节点的方法
    void addNode(uint32_t id, AttrNodeVNR node_attribute);

    // 添加有向边的方法
    void addEdge(uint32_t source, uint32_t target, AttrLinkVNR edge_attribute);

    // 获取图中的节点个数
    uint32_t getNodeNumber();

    // 获取图中的链路(有向边)个数
    uint32_t getEdgeNumber();

    // 获取 (from_node_index_vector, to_node_index_vector)
    std::tuple<std::vector<uint32_t>, std::vector<uint32_t>> getFromToNodeIndex();

    // 获取节点属性的方法
    AttrNodeVNR getNodeAttribute(uint32_t id);

    // 设置节点属性的方法
    void setNodeAttribute(uint32_t id, AttrNodeVNR node_attribute);

    // 获取边属性的方法
    AttrLinkVNR getEdgeWeight(uint32_t source, uint32_t target);

    // 设置边属性的方法
    void setEdgeWeight(uint32_t source, uint32_t target, AttrLinkVNR edge_attribute);

    // // 打印图信息的方法
    // void printGraphVNR() {
    //   for (const auto& pair : adjList) {
    //     const NodeVNR& node = pair.second;
    //     std::cout << "NodeVNR " << node.id << ": Attribute = " << node.node_attribute;
    //     std::cout << ", Out Edges: ";
    //     for (const EdgeVNR& edge : node.outEdges) {
    //       std::cout << "(" << edge.target << ", " << edge.edge_attribute << ") ";
    //     }
    //     std::cout << std::endl;
    //   }
    // }
};

//##################################################### 网络
struct NodeAttr {
  double D_R1; //节点中，资源类型R1的资源总量
  double D_R1_res; //节点中，资源类型R1的剩余容量
  double D_R1_unit_cost; // 在节点中，使用资源R1的单位cost
  double D_R2; //节点中，资源类型R1的资源总量
  double D_R2_res; //节点中，资源类型R1的剩余容量
  double D_R2_unit_cost; // 在节点中，使用资源R1的单位cost
};

struct EdgeConfig {
  uint32_t node_i;
  uint32_t node_j;
  std::string data_rate;
  std::string delay;
};

struct LinkAttr {
  double B; //网络链路的带宽总量
  double B_res; //网络链路带宽资源的剩余容量
  double B_unit_cost; // 在link中，使用带宽资源的单位cost，注意这里所有的link设置的 B_unit_cost 是相同的
};

struct NodeStepCache {
  uint32_t step_id;
  double res_time_R1; // 对于资源R1的剩余占用时间
  double occupy_D_R1; // 对于资源R1的占用量
  double res_time_R2; // 对于资源R2的剩余占用时间
  double occupy_D_R2; // 对于资源R2的占用量
  // 暂认为res_time_R1与res_time_R2相等

  NodeStepCache() = default;
  NodeStepCache(uint32_t s_i, double rtR1, double oDR1, double rtR2, double oDR2)
  : step_id(s_i), res_time_R1(rtR1), occupy_D_R1(oDR1), res_time_R2(rtR2) , occupy_D_R2 (oDR2){}
};

// void
// TraceNetStats (std::string context, Ptr<const Packet> macTxTrace)
// {
//   //MacTx挂钩：Trace source indicating a packet has arrived for transmission by this device.
//   uint32_t packet_size = macTxTrace->GetSize();//单位 byte
//   // Vector position = model->GetPosition ();
//   NS_LOG_UNCOND (context << " sizePacket = " << packet_size << ", y = " << packet_size);
// }

// csv数据集读取处理类
class DataPreprocess : public Object
{

  public:
    DataPreprocess ();
    virtual ~DataPreprocess ();
    static TypeId GetTypeId (void);

    // 数据存储字典
    std::vector<std::vector<std::string>> raw_data;
    // std::unique_ptr<std::vector<std::vector<std::string>>> raw_data_ptr;

    // 使用工作空间路径拼接
    std::string getFullPathByWorkSpace(std::string file_path);

    // 使用当前文件路径拼接
    std::string getCurrentPath(std::string file_path);

    // 自动尝试读取路径
    std::vector<std::vector<std::string>> autoRead(std::string file_path);

    // 读取csv文件
    std::vector<std::vector<std::string>> readCsv(std::string file_path);

    // 展示读取的内容
    int showContent();

    // 读取数据集适配GraphVNR对象
    // raw_data: ['job_name', 'task_name', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'task_num', 'plan_bandwidth']
    // raw_topology: ['job_name', 'start', 'end']
    std::vector<GraphVNR> dataToGraphVNR(std::vector<std::vector<std::string>> raw_data, std::vector<std::vector<std::string>> raw_topology, int sampled_num, int type_num_VNR);

    // 打印GraphVNR序列
    void showGraphVNR(std::vector<GraphVNR> graphVNR_vector, int showVNRNum);

};

}


#endif
