



# CAAI-BDSC-Task1-Baseline

[CAAI第八届全国大数据与社会计算学术会议 (CAAI-BDSC 2023) 社交图谱链接预测 ](https://tianchi.aliyun.com/competition/entrance/532073/introduction) 

[任务一：社交图谱小样本场景链接预测](https://tianchi.aliyun.com/competition/entrance/532073/information) 基线方法

### 说明

该仓库提供了基于异质信息网络表征模型的链接预测实现代码，其中数据处理基本流程、提交文件格式等可供参赛队伍参考，所用的CompGCN-DistMult模型并不是专用于小样本问题的模型。

代码参考自[CompGCN模型的DGL官方实现](https://github.com/dmlc/dgl/tree/master/examples/pytorch/compGCN)。

### 使用

#### 环境安装

建议通过Anaconda专门创建一个环境，并安装以下第三方依赖。

```
python=3.6.10
pytorch=1.10.2
dgl=0.9.1
pandas
json
heapq
ordered_set
collections
```

若有条件，请安装`pytorch`和`dgl`的GPU版本。

#### 数据

以初赛为例，报名参赛后，在[阿里巴巴天池平台](https://tianchi.aliyun.com/competition/entrance/532073/information)下载初赛数据文件，并放到`./data/`文件夹。

```
event_info.json
user_info.json
source_event_preliminary_train_info.json
target_event_preliminary_train_info.json
target_event_preliminary_test_info.json
```

#### 运行

在Terminal运行下面的命令：

```
python main.py --lr 0.003 --num_workers 10 --gpu 0
```

参赛队伍可以根据机器条件确定`gpu`、`num_workers`等参数。

#### 输出和提交

模型文件`baseline_ckpt.pth`会输出到`./checkpoint/`文件夹。

初赛结果文件`preliminary_submission.json`会输出到`./output/`文件夹，可以直接提交到天池平台。

### 参考结果

|          | MRR@5  | HITS@5 |
| -------- |--------|--------|
| Baseline | 0.24646 | 0.34941 |

