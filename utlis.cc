
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <string>
#include <cassert>
#include <typeinfo>
#include <vector>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unistd.h>
#include <cmath>
#include <numeric>
#include <limits>
#include <unordered_set>

#include "utlis.h"
#include "ns3/object.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
// #include "ns3/csma-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"

// #include "tutorial-app.h"

// #include "ns3/opengym-module.h"
// #include "mygym.h"


using namespace ns3;


NetConfig::NetConfig(
ns3::NodeContainer& node_container,
ns3::PointToPointHelper& pointToPoint,
ns3::Ipv4AddressHelper& address,
uint32_t nodeID_i, uint32_t nodeID_j,
std::string device_attr, std::string channel_attr
)
{
  // node_container = NodeContainer (nodes.Get (nodeID_i), nodes.Get (nodeID_j));

  // PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue (device_attr));
  pointToPoint.SetChannelAttribute ("Delay", StringValue (channel_attr));

  // NetDeviceContainer devices;
  devices = pointToPoint.Install (node_container);

  // InternetStackHelper stack;
  // stack.Install (nodes);

  // Ipv4AddressHelper address;
  address_base_str = addToIPv4Address ("0.0.0.1", count_IP_assign);
  NS_LOG_UNCOND("--**--address_base: " << address_base_str);
  // Ipv4Address address_network_str = "10.1.0.0";
  // Ipv4Mask address_mask_str = "255.255.0.0";
  // const char* pabs = "0.0.0.3";
  // char address_base_char[] = "0.0.0.3";
  const char* ptr_address_base_char = address_base_str.c_str();
  address.SetBase ("10.1.0.0", "255.255.0.0", ptr_address_base_char);

  std::string tmp_ip_string = "10.1." + std::to_string(count_instant+1) + ".0";
  address.SetBase (Ipv4Address(tmp_ip_string.data()),"255.255.255.0","0.0.0.1");

  // Ipv4InterfaceContainer interfaces;
  interfaces = address.Assign (devices);

  // 更新 count
  // uint32_t num_devices = devices.GetN ();
  uint32_t num_interfaces = interfaces.GetN();
  count_IP_assign += num_interfaces;
  count_instant += 1;

  // NS_LOG_UNCOND("--num_devices: " << num_interfaces);
  NS_LOG_UNCOND("--count_IP_assign: " << count_IP_assign);
  NS_LOG_UNCOND("--count_instant: " << count_instant);
}


NetConfig::~NetConfig ()
{
  // NS_LOG_FUNCTION (this);
}

TypeId
NetConfig::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::NetConfig")
    .SetParent<Object> ()
    .SetGroupName ("CustomConfig")
    ;
  return tid;
}

// addToIPv4Address函数：IPv4字符串地址+整数，并处理进位
std::string
NetConfig::addToIPv4Address(const std::string& ip, uint32_t number) {
    std::istringstream iss(ip);
    std::string segment;
    std::vector<uint32_t> ipParts;

    // 解析IPv4地址的四个部分
    for (uint32_t i = 0; i < 4 && std::getline(iss, segment, '.'); ++i) {
        ipParts.push_back(std::stoi(segment));
    }

    // 检查IP地址是否有效
    if (ipParts.size() != 4) {
        throw std::invalid_argument("Invalid IPv4 address format");
    }

    // 处理加法运算
    uint32_t carry = number;
    for (uint32_t i = 3; i >= 0; --i) {
        uint32_t sum = ipParts[i] + carry;
        ipParts[i] = sum % 256; // 获取当前位的值
        carry = sum / 256;      // 计算进位
    }

    // 如果最高位有进位，则地址无效
    if (carry > 0) {
        throw std::overflow_error("Adding the number to the IP address resulted in an overflow");
    }

    // 构建新的IPv4地址字符串
    std::ostringstream oss;
    for (size_t i = 0; i < ipParts.size(); ++i) {
        if (i > 0) oss << ".";
        oss << ipParts[i];
    }

    return oss.str();
}

uint32_t NetConfig::count_instant = 0;
uint32_t NetConfig::count_IP_assign = 0;
std::string NetConfig::address_base_str = "0.0.0.1";



// /**
//  * Congestion window change callback
//  *
//  * \param oldCwnd Old congestion window.
//  * \param newCwnd New congestion window.
//  */
// static void
// CwndChange(uint32_t oldCwnd, uint32_t newCwnd)
// {
//     NS_LOG_UNCOND(Simulator::Now().GetSeconds() << "\t" << newCwnd);
// }

// /**
//  * Rx drop callback
//  *
//  * \param p The dropped packet.
//  */
// static void
// RxDrop(Ptr<const Packet> p)
// {
//     NS_LOG_UNCOND("RxDrop at " << Simulator::Now().GetSeconds());
// }

// ##############################其他功能函数
// 比较函数，用于比较两个字符串形式的数字
bool compareStringsByNumbers(const std::string& a, const std::string& b) {
    // 找到两个字符串中第一个下划线的位置
    size_t underscoreA = a.find('_');
    size_t underscoreB = b.find('_');

    // 确保两个字符串都有下划线，并且下划线不是字符串的最后一个字符
    if (underscoreA == std::string::npos || underscoreB == std::string::npos ||
        underscoreA == a.size() - 1 || underscoreB == b.size() - 1) {
        return a < b; // 如果格式不正确，则按字典序排序
    }

    // 从第一个下划线前提取数字1
    std::istringstream issA(a.substr(0, underscoreA));
    std::istringstream issB(b.substr(0, underscoreB));
    int num1A, num1B;
    issA >> num1A;
    issB >> num1B;

    // 如果数字1不同，则直接比较数字1
    if (num1A != num1B) {
        return num1A < num1B;
    }

    // 从第一个下划线后提取数字2
    std::istringstream issA2(a.substr(underscoreA + 1));
    std::istringstream issB2(b.substr(underscoreB + 1));
    int num2A, num2B;
    issA2 >> num2A;
    issB2 >> num2B;

    // 比较数字2
    return num2A < num2B;
}

double mean(const std::vector<double>& data) {
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double variance(const std::vector<double>& data, double meanVal) {
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += std::pow(data[i] - meanVal, 2);
    }
    return sum / (data.size() - 1); // 使用样本方差的无偏估计
}

double skewness(const std::vector<double>& data) {
    double n = data.size();
    if (n < 3) {
        throw std::invalid_argument("The skewness error. Data set must contain at least 3 elements to compute skewness.");
    }

    double meanVal = mean(data);
    double var = variance(data, meanVal);
    if (var == 0) {
        return 0; // 避免除以零
    }

    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += std::pow(data[i] - meanVal, 3);
    }
    double skew = n * sum / ((n - 1) * (n - 2) * std::pow(var, 1.5));
    return skew;
}

double standardDeviation(const std::vector<double>& data) {
    double meanVal = mean(data);
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += std::pow(data[i] - meanVal, 2);
    }
    double variance = sum / (data.size() - 1); // 使用样本方差的无偏估计
    return std::sqrt(variance);
}

double calculateDuplicationRate(const std::vector<uint32_t> &vec) {

    // NS_LOG_UNCOND("==>> *******==>> uniqueCount = " << uniqueCount << "; totalCount = " << totalCount);

    if(vec.empty()) {
        return 0.0; // 空向量定义重复率为0
    }

    std::unordered_set<uint32_t> uniqueElements(vec.begin(), vec.end()); // 使用unordered_set去重，得到不同元素的数量
    uint32_t uniqueCount = uniqueElements.size(); // 不同元素的数量
    uint32_t totalCount = vec.size(); // 总元素数量

    // 计算重复率
    double duplicationRate = 1.0 - (static_cast<double>(uniqueCount) / totalCount);
    std::cout <<"==>> *******==>> uniqueCount = " << uniqueCount << "; totalCount = " << totalCount << std::endl;

    return duplicationRate;
}


//##################################### 任务VNR，以一个有向无环图的形式存储一个任务VNR的信息

// 图类，使用邻接表存储一个有向无环图
GraphVNR::GraphVNR ()
{
  // NS_LOG_FUNCTION (this);
}

GraphVNR::~GraphVNR ()
{
  // NS_LOG_FUNCTION (this);
}

TypeId
GraphVNR::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GraphVNR")
    .SetParent<Object> ()
    .SetGroupName ("CustomGraphVNR")
    ;
  return tid;
}

// 添加节点的方法
void
GraphVNR::addNode(uint32_t id, AttrNodeVNR node_attribute) {
  // NodeVNR node_struct(id, node_attribute);
  adjList[id] = NodeVNR(id, node_attribute);
}

// 添加有向边的方法
void
GraphVNR::addEdge(uint32_t source, uint32_t target, AttrLinkVNR edge_attribute) {
  if (adjList.find(source) != adjList.end()) { //在addEdge函数中，这个条件用于确保在尝试向图中添加边之前，源节点（source）已经存在于邻接表中。如果不存在，则不会添加边，这可以避免向图中添加无效的边（即源节点不存在的边）。
    adjList[source].addOutEdge(target, edge_attribute);
  }
}

// 获取图中的节点个数
uint32_t
GraphVNR::getNodeNumber() {
  uint32_t n_nodes_VNR = static_cast<uint32_t> (adjList.size());
  return n_nodes_VNR;
}

// 获取图中的链路(有向边)个数
uint32_t
GraphVNR::getEdgeNumber() {
  uint32_t n_links_VNR = 0;
  for (const auto& pair : adjList) {
    n_links_VNR = n_links_VNR + static_cast<uint32_t>(pair.second.outEdges.size());
  }
  return n_links_VNR;
}

// 获取 (from_node_index_vector, to_node_index_vector)
std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>
GraphVNR::getFromToNodeIndex() {
  std::vector<uint32_t> from_node_index_vector;
  std::vector<uint32_t> to_node_index_vector;
  std::vector<uint32_t> nosort_t_n_i_vector;
  for (const auto& pair : adjList) {
    for (const auto& value : pair.second.outEdges) {
      from_node_index_vector.push_back(pair.first);
      nosort_t_n_i_vector.push_back(value.target);
    }
    std::sort(nosort_t_n_i_vector.begin(), nosort_t_n_i_vector.end());
    // 将 nosort_t_n_i_vector 的所有元素插入到 to_node_index_vector 的末尾
    to_node_index_vector.insert(to_node_index_vector.end(), nosort_t_n_i_vector.begin(), nosort_t_n_i_vector.end());
    // for (const auto& value : nosort_t_n_i_vector) {
    //   to_node_index_vector.push_back(value.target);
    // }
    nosort_t_n_i_vector.clear();
  }
  return std::make_tuple(from_node_index_vector, to_node_index_vector);
}

// 获取节点属性的方法
AttrNodeVNR
GraphVNR::getNodeAttribute(uint32_t id) {
  if (adjList.find(id) != adjList.end()) {
    return adjList[id].node_attribute;
  }
  return AttrNodeVNR(-1,-1,-1); // 节点不存在时返回的错误码
}

// 设置节点属性的方法
void
GraphVNR::setNodeAttribute(uint32_t id, AttrNodeVNR node_attribute) {
  if (adjList.find(id) != adjList.end()) {
    adjList[id].node_attribute = node_attribute;
  }
}

// 获取边属性的方法
AttrLinkVNR
GraphVNR::getEdgeWeight(uint32_t source, uint32_t target) {
  if (adjList.find(source) != adjList.end()) {
    for (const EdgeVNR& edge : adjList[source].outEdges) {
      if (edge.target == target) {
        return edge.edge_attribute;
      }
    }
  }
  //return -1; // 边不存在时返回的错误码
}

// 设置边属性的方法
void
GraphVNR::setEdgeWeight(uint32_t source, uint32_t target, AttrLinkVNR edge_attribute) {
  if (adjList.find(source) != adjList.end()) {
    for (EdgeVNR& edge : adjList[source].outEdges) {
      if (edge.target == target) {
        edge.edge_attribute = edge_attribute;
        return;
      }
    }
  }
  // 如果边不存在，可以选择添加它或不做任何操作
}


/*
* 数据预处理部分
*/
DataPreprocess::DataPreprocess ()
{
  // NS_LOG_FUNCTION (this);
}

DataPreprocess::~DataPreprocess ()
{
  // NS_LOG_FUNCTION (this);
}

TypeId
DataPreprocess::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::DataPreprocess")
    .SetParent<Object> ()
    .SetGroupName ("CustomDataPreprocess")
    ;
  return tid;
}

// 使用工作空间路径拼接
std::string
DataPreprocess::getFullPathByWorkSpace(std::string file_path) {
  const int size = 65535;
  char buf[size];
  if(getcwd(buf, sizeof(buf)) != nullptr) {
      std::cout << "current working directory: " << buf << std::endl;
  }
  // 在当前工作路径下拼接/
  std::strcat(buf, "/");
  // 拼接传入路径
  std::strcat(buf, file_path.c_str());
  std::cout << "current working directory: " << buf << std::endl;
    return buf;
}

// 使用当前文件路径拼接
std::string
DataPreprocess::getCurrentPath(std::string file_path) {
  std::string currentPath = __FILE__;
  size_t lastSlashIndex = currentPath.find_last_of("/\\");
  if(lastSlashIndex != std::string::npos) {
    currentPath = currentPath.substr(0, lastSlashIndex);
  } else {
    currentPath = ".";
  }
  // 在当前工作路径下拼接/
  currentPath += "/" + file_path;
  return currentPath;
}

// 读取csv文件方法
std::vector<std::vector<std::string>>
DataPreprocess::autoRead(std::string file_path) {

  // 是否使用绝对路径打开
  bool open_current_flag = false;
  // 1.直接打开文件
  std::cout << "使用工作空间路径打开文件: " << file_path << std::endl;
  std::ifstream file(file_path);
  if(!file.is_open()) {
    std::cout << "文件打开失败: " << file_path << std::endl;
    file.close();
    open_current_flag = true;
  } else {
    std::cout << "文件打开成功!: " << file_path << std::endl;
  }

  if(open_current_flag) {
    // 2. 尝试使用绝对路径读取传入文件
    std::cout << "使用绝对路径尝试打开文件：" << file_path << std::endl;
    // 获取绝对路径
    file_path = getCurrentPath(file_path);
    std::ifstream file(file_path);
    if(!file.is_open()) {
      std::cout << "使用绝对路径文件打开失败: " << file_path << std::endl;
      return {{}};
    } else {
      std::cout << "使用绝对路径文件打开成功!: " << file_path << std::endl;
    }
  }

  return readCsv(file_path);
}

// 读取csv文件方法
std::vector<std::vector<std::string>>
DataPreprocess::readCsv(std::string file_path) {
  std::ifstream file(file_path);
  // 按行读取数据
  std::string line;
  while(std::getline(file, line)) {
    // 记录每行数据
    std::vector<std::string> row;

    // 分割字符串，将每个字段存到row中
    size_t pos = 0;
    std::string token;
    while((pos = line.find(",")) != std::string::npos) {
      // 找到分割符位置并截取
      token = line.substr(0, pos);
      // 将截取字符存入row中
      row.push_back(token);

      // 将line起始位置设为分割符下一位置，继续查找下一分割符
      line.erase(0, pos+1);
    }

    // 将剩下的最后一个字段存到row中
    row.push_back(line);

    // 将row保存到存储字典中
    raw_data.push_back(row);

  }
  return raw_data;
}

// 展示读取的内容
int
DataPreprocess::showContent() {
  // 输出raw_data内容
  uint32_t num_show_row = 20;
  for (uint32_t i = 0; i < num_show_row; i++) {
    std::vector<std::string> row = raw_data[i];
    for(const auto& col : row) {
      std::cout << col << " ";
    }
    std::cout << std::endl;
  }

  // for(const auto& row : input_read) {
  //   for(const auto& col : row) {
  //     std::cout << col << " ";
  //   }
  //   std::cout << std::endl;
  // }
  return 0;
}

// 读取数据集适配GraphVNR对象
/*
* raw_data: ['job_name', 'task_name', 'start_time', 'end_time', 'plan_cpu', 'plan_mem', 'task_num', 'plan_bandwidth']
* raw_topology: ['job_name', 'start', 'end']
*
*/
std::vector<GraphVNR>
DataPreprocess::dataToGraphVNR(std::vector<std::vector<std::string>> raw_data, std::vector<std::vector<std::string>> raw_topology, int sampled_num, int type_num_VNR) {
  std::cout << "------------------------------- 方法进入成功 ----------------------------------------------" << std::endl;

  //根据job记录VNR信息及当前VNR内VNF数量
  std::map<std::string, GraphVNR*> graphVNR_map;
  std::map<std::string, uint32_t> graphVNRCnt_map;
  // 读取所有数据 - 初始化VNR列表并根据raw_data添加节点
  for(uint32_t i = 0; i < raw_data.size(); i++) {
    std::vector<std::string> row_i = raw_data[i];
    std::string job_name = row_i[0];
    uint32_t rand_VNR_type = static_cast<uint32_t>(rand() % (type_num_VNR));
    //判断键是否存在 - 已有GraphVNR对象，直接添加节点; 没有GraphVNR对象，添加对象并初始化统计表
    if(graphVNR_map.count(job_name)) {
      (*graphVNR_map[job_name]).addNode(graphVNRCnt_map[job_name], AttrNodeVNR(std::stod(row_i[4]) / 100, std::stod(row_i[5]), rand_VNR_type));
      graphVNRCnt_map[job_name] += 1;
    } else {
      graphVNR_map[job_name] = new GraphVNR();
      (*graphVNR_map[job_name]).addNode(0, AttrNodeVNR(std::stod(row_i[4]) / 100, std::stod(row_i[5]), rand_VNR_type));
      graphVNRCnt_map[job_name] = 1;
    }
  }

  // 根据VNR列表及raw_topoloty添加连边
  for(uint32_t j = 0; j < raw_topology.size(); j++) {
    std::vector<std::string> row_j = raw_topology[j];
    std::string job_name = row_j[0];
    (*graphVNR_map[job_name]).addEdge(static_cast<uint32_t>(std::stoi(row_j[1])), static_cast<uint32_t>(std::stoi(row_j[2])), AttrLinkVNR(std::stod(row_j[3])));
  }

  // graphVNR_map转graphVNR_vector
  std::vector<GraphVNR> graphVNR_vector;
  for(const auto& kv : graphVNR_map) {
    graphVNR_vector.push_back(*kv.second);
  }

  int showVNRNum = sampled_num;
  showGraphVNR(graphVNR_vector, showVNRNum);

  std::cout << "---------------------------- 返回主函数 -----------------------------------------" << "VNR数量: " << graphVNR_vector.size() << std::endl;

  return graphVNR_vector;
}

// 读取数据集适配GraphVNR对象
void
DataPreprocess::showGraphVNR(std::vector<GraphVNR> graphVNR_vector, int showVNRNum) {
  int cnt = 0;
  for(auto graphVNR : graphVNR_vector) {
    for (const auto& pair : graphVNR.adjList) {
        const NodeVNR& node = pair.second;
        std::cout << "NodeVNR " << node.id << ": Attribute = " << node.node_attribute.d_R1;
        std::cout << ", Out Edges: ";
        for (const EdgeVNR& edge : node.outEdges) {
          std::cout << "(" << edge.target << ", " << edge.edge_attribute.b_e << ") ";
        }
        std::cout << std::endl;
      }

    if(cnt == showVNRNum) break;
    cnt++;
  }

}
