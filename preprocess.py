import pandas as pd
import numpy as np

import random
import networkx as nx
import matplotlib.pyplot as plt


import networkx as nx
import random
import time

def generate_topology(df, bandwidth_config):

    cnt_df = df['job_name'].value_counts().reset_index()
    cnt_df.columns = ['job_name', 'task_cnt']

    # 使用列表存储提升速度用于转化至dataframe
    topology_list = []
    # 遍历job信息，为每个job(VNR)生成随即拓扑
    for index, row in cnt_df.iterrows():
        cdrg = ConnectedDirectedRandomGraph(row['job_name'], row['task_cnt'])
        cdrg.generate_connected_directed_random_graph()
        cdrg.to_sequential_directed_random_graph_by_array()
        cdrg.to_obj_list(bandwidth_config)
        # cdrg.print_graph()
        # 使用列表方式减少拼接提升速度
        topology_list.extend(cdrg.single_graph_list)

    topology_df = pd.DataFrame(topology_list)
    #print(topology_df.head(10))
    return topology_df

class ConnectedDirectedRandomGraph:
    def __init__(self, id, num_nodes):
        self.idx = id
        self.num_nodes = num_nodes
        self.graph = {node: [] for node in range(num_nodes)}
        self.visited = [False] * num_nodes
        # 备份一个字典列表用于转换为dataframe进行存取
        self.single_graph_list = []

    def add_edge(self, from_node, to_node):
        # 添加有向边到图中
        self.graph[from_node].append(to_node)
        # 准备写入信息 - 如使用by_list方法需打开此处注释
        #self.single_graph_list.append({'job_name': self.idx, 'start': from_node, 'end': to_node, 'plan_bandwidth': round(np.random.uniform(10, 100), 1)})

    def dfs(self, node):
        # 深度优先搜索来标记访问过的节点
        self.visited[node] = True
        for neighbor in self.graph[node]:
            if not self.visited[neighbor]:
                self.dfs(neighbor)

    def generate_connected_directed_random_graph(self):
        # 从第一个节点开始，构建连通图
        self.visited[0] = True
        for _ in range(self.num_nodes - 1):
            # 随机选择一个未访问过的节点作为起始节点
            start_node = random.choice([node for node, visited in enumerate(self.visited) if not visited])
            # 随机选择一个已访问过的节点作为目标节点
            end_node = random.choice([node for node, visited in enumerate(self.visited) if visited])
            # 添加从目标节点到起始节点的边
            self.add_edge(end_node, start_node)
            #self.add_edge(start_node, end_node)
            # 标记起始节点为已访问
            self.visited[start_node] = True

        # 确保图是连通的，通过从每个未访问的节点开始DFS，并添加边到已访问的节点
        for node in range(self.num_nodes):
            if not self.visited[node]:
                # 找到一个已访问的节点来添加边
                visited_nodes = [n for n, v in enumerate(self.visited) if v]
                if not visited_nodes:
                    raise ValueError("Unable to generate a connected graph.")
                end_node = random.choice(visited_nodes)
                self.add_edge(node, end_node)
                self.dfs(node)  # 从这个新访问的节点开始DFS，确保所有可达节点都被访问

     # 检验是否有无序部分 - 通过graph[]实现
    def check_sequential_DAG_by_array(self):
        for node in range(self.num_nodes):
            for to_node in self.graph[node]:
                if to_node <= node:
                    return False, node, to_node
        return True, 0, 0

    # 将随机DAG图转换为顺序DAG图 - 通过graph[]实现
    def to_sequential_directed_random_graph_by_array(self):
        self.single_graph_list.sort(key=lambda k : (k.get('end', 0)), reverse=True)

        # 满足顺序DAG时退出循环
        completeFlag = False
        while not completeFlag:
            completeFlag, from_original, to_original = self.check_sequential_DAG_by_array()
            #print("flag: ", completeFlag, "; start: ", from_original, " ; end: ", to_original)
            #time.sleep(3)
            # 找到可行图，打印并退出
            if completeFlag:
                # for item in self.single_graph_list:
                #     print(item)
                break
            # 交换起止节点边集
            self.graph[from_original], self.graph[to_original] = self.graph[to_original], self.graph[from_original]
            # 将边中端节点为起止节点的交换位置
            for node in range(self.num_nodes):
                for to_node in self.graph[node]:
                    if from_original == to_node:
                        self.graph[node][self.graph[node].index(to_node)] = to_original
                    if to_original == to_node:
                        self.graph[node][self.graph[node].index(to_node)] = from_original

    # 将graph数组转换为对象列表供转化为dataframe存储
    def to_obj_list(self, bandwidth_config):
        # 拆解传参
        [band_low, band_high] = bandwidth_config
        for node in range(self.num_nodes):
            for to_node in self.graph[node]:
                self.single_graph_list.append({'job_name': self.idx, 'start': node, 'end': to_node, 'plan_bandwidth': round(np.random.uniform(band_low, band_high), 1)})

    # 检验是否有无序部分 - 通过single_graph_list实现 （未使用
    def check_sequential_DAG_by_list(self):
        for edge in self.single_graph_list:
            if edge['start'] >= edge['end']:
                return False, edge['start'], edge['end']

        return True, 0, 0

    # 将随机DAG图转换为顺序DAG图 - 通过single_graph_list实现 （未使用
    def to_sequential_directed_random_graph_by_list(self):
        self.single_graph_list.sort(key=lambda k : (k.get('end', 0)), reverse=True)
        for item in self.single_graph_list:
            print(item)

        completeFlag = False
        while not completeFlag:
            completeFlag, from_original, to_original = self.check_sequential_DAG_by_list()

            # print("flag: ", completeFlag, "; start: ", from_original, " ; end: ", to_original)
            if completeFlag:
                for item in self.single_graph_list:
                    print(item)
                break
            for edge in self.single_graph_list: # 依次交换起始节点编号
                if from_original ==  edge['start']:
                    edge['start'] = to_original
                else:
                    if to_original ==  edge['start']:
                        edge['start'] = from_original
                if from_original ==  edge['end']:
                    edge['end'] = to_original
                else:
                    if to_original ==  edge['end']:
                        edge['end'] = from_original


    def print_graph(self):
        G = nx.DiGraph()
        for node in range(self.num_nodes):
            G.add_node(node)
            print(f"Node {node}: {self.graph[node]}")
            for neighbor in self.graph[node]:
                G.add_edge(node, neighbor)
        nx.draw(G, cmap=plt.get_cmap('jet'), with_labels=True, font_weight='bold')
        plt.show()



# ---------------------------------------------------------------------- 主函数 -----------------------------------------------------------------------------------

# 每个job中task数量的上下限
task_num_low = 5
task_num_high = 8

# 均匀分布中带宽的上下界 单位: bps
band_low = 1000
band_high = 2000

# 均匀分布中CPU的上下界
cpu_low = 50
cpu_high = 100

# 均匀分布中内存的上下界
mem_low = 0.2
mem_high = 0.59

print("----------------------------------------- Start preparing dataset -----------------------------------------------")
index_name=['task_name', 'inst_num', 'job_name', 'task_type', 'status', 'start_time', 'end_time' ,  'plan_cpu', 'plan_mem']
df = pd.read_csv('dataset/raw/batch_task.csv', header=None, names=index_name)
# 过滤非terminated的值
df = df[df['status'] == 'Terminated']
# 只获取task_name和job_name
df2=df[['job_name', 'task_name', 'start_time', 'end_time', 'plan_cpu', 'plan_mem']]

print("----------------------------------------- Gathering task information -----------------------------------------------")
# 按照job分组统计任务数量
df2['task_num'] = df2.groupby('job_name')['task_name'].transform('count')


# 筛选在节点范围内的数据
df2 = df2[df2['task_num'].between(task_num_low, task_num_high)]

# 填充所需带宽信息
df2['plan_bandwidth'] = np.random.uniform(band_low, band_high, size=len(df2))

# 填充cpu和内存请求空值
df2.fillna(value=np.random.uniform(cpu_low, cpu_high))
df2.fillna(value=np.random.uniform(mem_low, mem_high))

# 保留小数位数
df2 = df2.round({'plan_cpu': 1, 'plan_mem': 1, 'plan_bandwidth': 1})

# 重置索引
df2.reset_index(drop=True, inplace=True)
#print(df2.head(10))

# index_name_cleaned=['task_name', 'job_name', 'start_time', 'end_time', 'task_num', 'plan_cpu', , 'plan_mem', 'plan_bandwidth']
# 把清洗出来的数据保存起来,index=0不保留行索引 ,head=None不保存列名
#df2.to_csv('dataset/generated_files/batch_task_cleaned.csv', mode='w', header=None, index=0)

print("不同任务数量：", df2['job_name'].nunique())

print("----------------------------------------- Generating topology for each job -----------------------------------------------")
# 采样10个job ------------- test
job_list = df2['job_name'].unique()[0:1000]
df3 = df2[df2['job_name'].isin(job_list)]
bandwidth_config = [band_low, band_high]
save_topology_df = generate_topology(df3, bandwidth_config)

index_name_cleaned=['job_name', 'task_name', 'start_time', 'end_time', 'task_num', 'plan_cpu', 'plan_bandwidth']
# 把清洗出来的数据保存起来,index=0不保留行索引 ,head=None不保存列名
df3.to_csv('dataset/generated_files/batch_task_cleaned.csv', mode='w', header=None, index=0)
topology_index_name_cleaned=['job_name', 'start', 'end']
save_topology_df.to_csv('dataset/generated_files/generated_topology.csv', mode='w', header=None, index=0)
print("----------------------------------------- End preparing dataset -----------------------------------------------")
