# Graph Pointer Network Assisted Deep Reinforcement Learning (GPRL)

This repository contains the official implementation of GPRL, as presented in the paper "Graph Pointer Network Assisted Deep Reinforcement Learning (GPRL) for VNE in dynamic network environments".

## Overview

 GPRL is an automatic VNE method based on graph pointer network and Deep
 Reinforcement Learning (DRL). By combining the graph neural network and pointer network, we
 design a novel graph pointer network as the DRL agent. It employs the graph attention network (GAT)
 to encode graph feature data and decodes to output the embedding strategy via the pointer network architecture. 
 Meanwhile, the Proximal Policy Optimization (PPO) algorithm is used to effectively train
 the designed agent. The effectiveness and superiority of GPRL is verified by simulation experiments.

<div align=center>
    <img src="/pictures/framework.png" alt="HL-GNN" width="60%" height="60%">
</div>


## Data Processing Description
We use the dataset for the VNE problem from the [Alibaba cluster dataset](https://github.com/alibaba/clusterdata). 
The dataset is derived from a sampling of a production cluster at Alibaba (4000 devices for 8 days). 
It records online services (also known as Long-Running Applications, LRA) and batch workloads on each machine in the cluster. 
The dataset is 270GB in size and contains six files when fully unzipped. 
We build the VNR graph using the relationship between jobs and tasks, and this part of the information is recorded in the `batch task` file, 
which records information about the instances in the batch workload, that is, the DAG information of each job. 
Therefore, we only deal with the `batch_task.csv` file, which is about 800MB in size. 
In this setting, each job corresponds to multiple tasks, and each task has the requested amount of CPU and memory resources, 
and the `task_name` defines the orientation relation to the different tasks within the job.

In the construction of VNRs, we use a job to refer to a complete VNR, and the DAG graph is constructed according to the task nodes contained in the job, 
and the execution order of different tasks within a job constitutes a set of edges. 
In addition, we kept the original `job_name`, `task_name`, `start_time`, `end_time`, `plan_CPU`, and `plan_mem` in the dataset. 
The `plan_bandwidth` attribute is generated for it to simulate the demand for bandwidth resources. 
To simulate different VNFs, we randomly generate the `VNF_type` field for tasks to identify different VNF functions. 
We screen out the job instances containing 5 to 8 nodes to build the DAG graph for VNR from a large number of job instances. 
These instances retain the original starting and ending time of the task and the amount of CPU and memory resources occupied (missing values are supplemented according to the uniform distribution). 
Then we fill the requested bandwidth resources according to the uniform distribution. These characteristics constitute the key parameters of our simulations.

For the underlying physical network, we use different real topologies to complete the construction of different physical networks, 
including GEANT, Abilene, ChinaNet, and Xspedius. 
After creating the physical network topology, we set a relatively uniform amount of resources for each node and link, and randomly assign the cost per unit resource usage to each node to distinguish the capabilities of different nodes. 
In addition, for each node, we also set the carrying capacity of different VNF types for it. 
In the experiment, we assumed five different VNF types. 
Thus, for the topology, each node randomly runs a different number and type of VNF.

Attached project introduction: <br>
the Alibaba Cluster Trace Program is published by Alibaba Group. 
By providing cluster trace from real production, the program helps the researchers, students and people who are interested in the field to get better understanding of the characteristics of modern internet data centers (IDC's) and the workloads. 

Attached dataset introduction: <br>
the cluster-trace-v2018 includes about 4000 machines in a period of 8 days. Besides having a larger scaler than trace-v2017, this piece trace also contains the DAG information of our production batch workloads.



## Requirements
- python >= 3.10 <br>
- [ns-3.39](https://www.nsnam.org/releases/ns-allinone-3.39.tar.bz2) <br>
- [ns3-gym](https://github.com/tkn-tub/ns3-gym)

## Installation
After installing the environment according to the installation guidelines of these projects, 
it's time to install this project.

Clone the repository and install the necessary dependencies:

```bash
cd YOUR_NS3_DIRECTORY
cd contrib/opengym/examples/
git clone https://github.com/ni4555/GPRL.git
cd GPRL
pip install -r requirements.txt
```

## Usage

### Alibaba Cluster Datasets Preprocess
Our VNR problem is built on the [Alibaba cluster dataset](https://github.com/alibaba/clusterdata), and the experimental setup can be found in the paper.
The fully decompressed dataset has a size of 270GB, but since we only used the batch_task file, the size is approximately 800MB.
After you receive the data file, please place it in the following directory:

```bash
cd dataset/raw/
PASTE_BATCH_TASK_CSV_HERE
cd ../../
```

Because we only need the correspondence and related features of tasks and jobs, and generate 
some resource requests and lifecycles for each request, we need to carry out data preprocessing. 
Please run the following command to complete the data preprocessing.
After running the preprocessing script, the cleaned data file `batch_task_clean.csv` and 
the generated topology file `generated-topology.csv` based on this data will appear in the following path (`/dataset/generated_files/`).


```bash
python preprocess.py
```

In this preprocessing setting, we have retained `task_name` for each task, `job_name`, `plan_cpu`, `plan_mem`, 
`start_time`, `end_time`， And generated bandwidth requirements (`plan_bandwidth`) for each edge according to a uniform distribution.
Meanwhile, you can modify the generation range of key attributes such as CPU, memory (only if that unit is empty in the dataset), 
bandwidth (generate for each task), etc. by adjusting the parameter settings in lines 173-186 of the script above.
And control the number of sampled jobs through line 226, and control or add the task attributes used through line 231.


### Build Module
This project is consistent with other demos in ns3-gym and is a module in ns-3. Therefore, before running the simulation, 
it is necessary to first build the module into the ns-3 project.

```bash
cd YOUR_NS3_DIRECTORY
./ns3 clean
./ns3 configure --build-profile=optimized --enable-examples --enable-tests --enable-python-bindings
./ns3 build
```

### Start Simulation

After the data preparation is completed, we can simply simulate it by invoking the script below. 

```bash
cd contrib/opengym/examples/GPRL/
./test.py
```

### After Simulation

After the training process is completed, a new result directory will be automatically created and important process data from the training process will be saved.
For the reinforcement learning reward changes during the training process, you can visualize them by running the following script.

```bash 
python viewResult.py
```

Due to the variable file name generated, it's recommended to check the file name under path `/result/reward_save/` and 
modify the reward file name in line 5 of file `viewResult.py` according to that directory.


### Other Settings

#### Algorithm Setting
Some settings about the algorithm can be modified through `test.py`. The 82 line total steps can 
control the total number of experimental steps performed in reinforcement learning. 
By modifying the parameters from line 96 to line 108, important parameters during the learning process can be controlled.


#### Topology Setting
We also conducted simulations on the settings of various topologies. You can turn on or off the function in `sim.cc`.
In these settings, `edges` control the bandwidth and latency of connections;
`node_attr_vector` records the total amount of resources, remaining amount, and unit resource cost; 
`node_VNF_deployment_vector` controls the types of VNFs that a node can accept.


#### Reward Setting
If you want to modify the simulation process in the experiment, such as the 
calculation process of reinforcement learning rewards in the experiment and the resource utilization process, 
we recommend that you carefully read the `mygym.cc` file and try to make your own custom modifications.


#### VNR Setting
If you want to expand the number of VNF types, you can achieve this by adjusting the `m_n_type_VNF` attribute on line 96 of the `mygym.cc` file.

If you want to filter the DAG graph of VNRs with a specified number of nodes, you can do so by modifying lines 97, 98 of the `mygym.cc` file with `m_n_min_VNF` and `m_n_max_VNF`.


## Results

The comparison of the performance of GPRL on GEANT network with four different baseline models on the following four different performance indexes is shown in the figure below.
It can be seen that our method can provide higher service acceptance rates, long-term average revenue, 
and reduce long-term average costs, increasing the revenue-cost ratio by more than 24%.
<div align=center>
    <img src="/pictures/acceptance_ratio_revenue_cost.png" alt="HL-GNN" width="60%" height="60%">
</div>

In addition, you can also observe various performance indicators of the network by utilizing different network topologies. 
The performance on different topologies is shown in the figure below.
<div align=center>
    <img src="/pictures/acceptance_ratio_revenue_cost_more_networks.png" alt="HL-GNN" width="60%" height="60%">
</div>

More experiment results can be found in our paper.

Feel free to reach out if you have any questions! [✉️](guoshuhan@mail.tsinghua.edu.cn)
