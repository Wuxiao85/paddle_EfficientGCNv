# 基于PaddlePaddle实现的 EfficientGCNv1
## 1. 简介
[EfficientGCN: Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition](https://paperswithcode.com/paper/constructing-stronger-and-faster-baselines)
一文提出了基于骨架行为识别的baseline，在论文中，将基于骨架识别的网络分为input branch和 main stream两部分。Input branch 用于提取骨架数据的多模态特征，提取的特征通过concat等操作完成特征融合后将输入main stream中预测动作分类。

![EfficientGCN](./images/model.PNG)

[官方源码](https://gitee.com/yfsong0709/EfficientGCNv1)
## 2. 数据集和复现精度
#### 2.1 NTU-RGB-D60
下载地址: https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H
#### 2.2 复现精度（待调整）
|   EfficientGCN-B0   | x-sub | x-view |
|:----:|:-----:|:------:| 
|  论文  | 90.2  |  94.9  | 
| 复现精度 | 90.22  | 94.71  |
## 3. 准备环境
**基本环境**
* Python: 3.7
* PaddlePaddle: 2.2.2

**模块安装**

    pip install -r requirements.txt
    
## 4. 快速开始
#### 4.1 下载数据集
到官网下载好数据集
#### 4.2 下载本项目及训练权重
    git clone git@github.com:Wuxiao85/paddle_EfficientGCNv.git
预训练网络下载: [xsub.pdparams](https://github.com/Wuxiao85/paddle_EfficientGCNv/blob/main/pretrain_model/xsub.pdparams), [xview.pdparams](https://github.com/Wuxiao85/paddle_EfficientGCNv/blob/main/pretrain_model/xview.pdparams)
#### 4.3 数据预处理
修改config文件中的骨架数据地址

    ntu60_path: <your file path to ntu60>
    ntu120_path: <your file path to ntu120>

运行
    
    python main.py -c 2001 -gd -np
    python main.py -c 2002 -gd -np
运行完成后的数据目录结构如下

    ├─data	
    │  ├─npy_dataset
    │  │  │─original   
    │  │  │  │─ ntu-xsub 
    │  │  │  │  │─ train_data.npy
    │  │  │  │  │─ train_label.pkl
    │  │  │  │  │─ eval_data.npy
    │  │  │  │  │─ eval_label.pkl
    │  │  │─transformed  
    │  │  │  │─ ntu-xview 
    │  │  │  │  │─ train_data.npy
    │  │  │  │  │─ train_label.pkl
    │  │  │  │  │─ eval_data.npy
    │  │  │  │  │─ eval_label.pkl
    │
    │
#### 4.4 模型训练
执行

    # export CUDA_VISIBLE_DEVICES=0 // 用到的gpu编号
    # python main.py -c 2001 // 在x-sub数据集上训练

    # export CUDA_VISIBLE_DEVICES=0 // 用到的gpu编号
    # python main.py -c 2002 // 在x-view数据集上训练

训练模型将保留在temp文件夹下。
训练log: [xview.log](https://github.com/Wuxiao85/paddle_EfficientGCNv/blob/main/workdir/2002_EfficientGCN-B0_ntu-xview/2022-06-05%2000-51-10/log.txt), [xsub.log](https://github.com/Wuxiao85/paddle_EfficientGCNv/blob/main/workdir/2001_EfficientGCN-B0_ntu-xsub/2022-06-05%2000-52-01/log.txt)


部分训练log：


x-view：
    
    [ 2022-06-06 05:42:24,943 ] Best top-1 accuracy: 94.71%, Total time: 00d-13h-15m-32s
    [ 2022-06-06 05:42:24,943 ] 
    [ 2022-06-06 05:53:16,878 ] Epoch: 70/70, Training accuracy: 37300/37632(99.12%), Training time: 651.93s
    [ 2022-06-06 05:53:16,879 ] 
    [ 2022-06-06 05:53:16,880 ] Evaluating for epoch 70/70 ...
    [ 2022-06-06 05:55:27,984 ] Top-1 accuracy: 17898/18928(94.56%), Top-5 accuracy: 18800/18928(99.32%), Mean loss:0.1896
    [ 2022-06-06 05:55:27,985 ] Evaluating time: 131.10s, Speed: 144.37 sequnces/(second*GPU)
    [ 2022-06-06 05:55:27,985 ] 
    [ 2022-06-06 05:55:27,985 ] Saving model for epoch 70/70 ...
    [ 2022-06-06 05:55:28,010 ] Best top-1 accuracy: 94.71%, Total time: 00d-13h-28m-35s
    [ 2022-06-06 05:55:28,010 ] 
    [ 2022-06-06 05:55:28,010 ] Finish training!
    [ 2022-06-06 05:55:28,010 ] 

    [ 2022-06-06 05:55:28,010 ] Best top-1 accuracy: 94.71%, Total time: 00d-13h-28m-35s
    [ 2022-06-06 05:55:28,010 ] 
    [ 2022-06-06 05:55:28,010 ] Finish training!
    [ 2022-06-06 05:55:28,010 ]  

x-sub:
    
    [ 2022-06-06 06:23:48,406 ] Evaluating for epoch 69/70 ...
    [ 2022-06-06 06:25:40,001 ] Top-1 accuracy: 14868/16480(90.22%), Top-5 accuracy: 16238/16480(98.53%), Mean loss:0.3653
    [ 2022-06-06 06:25:40,001 ] Evaluating time: 111.59s, Speed: 147.68 sequnces/(second*GPU)
    [ 2022-06-06 06:25:40,001 ] 
    [ 2022-06-06 06:25:40,001 ] Saving model for epoch 69/70 ...
    [ 2022-06-06 06:25:40,028 ] Best top-1 accuracy: 90.22%, Total time: 00d-13h-58m-48s
    [ 2022-06-06 06:25:40,028 ] 
    [ 2022-06-06 06:37:12,182 ] Epoch: 70/70, Training accuracy: 39904/40080(99.56%), Training time: 692.15s
    [ 2022-06-06 06:37:12,183 ] 
    [ 2022-06-06 06:37:12,183 ] Evaluating for epoch 70/70 ...
    [ 2022-06-06 06:39:03,073 ] Top-1 accuracy: 14840/16480(90.05%), Top-5 accuracy: 16249/16480(98.60%), Mean loss:0.3652
    [ 2022-06-06 06:39:03,073 ] Evaluating time: 110.89s, Speed: 148.62 sequnces/(second*GPU)
    [ 2022-06-06 06:39:03,073 ] 
    [ 2022-06-06 06:39:03,073 ] Saving model for epoch 70/70 ...
    [ 2022-06-06 06:39:03,098 ] Best top-1 accuracy: 90.22%, Total time: 00d-14h-12m-11s
    [ 2022-06-06 06:39:03,098 ] 
    [ 2022-06-06 06:39:03,099 ] Finish training!
    [ 2022-06-06 06:39:03,099 ] 
    [ 2022-06-06 06:39:03,098 ] Best top-1 accuracy: 90.22%, Total time: 00d-14h-12m-11s
    [ 2022-06-06 06:39:03,098 ] 
    [ 2022-06-06 06:39:03,099 ] Finish training!
    [ 2022-06-06 06:39:03,099 ]
    
#### 4.5 模型预测
执行

    # export CUDA_VISIBLE_DEVICES=0 // 用到的gpu编号
    # python main.py -c 2001 -e -pp <path to pretrain>// 在x-sub数据集上训练

    # export CUDA_VISIBLE_DEVICES=0 // 用到的gpu编号
    # python main.py -c 2002 // 在x-view数据集上训练


x-view 预测结果:
    
    INFO 2022-06-06 09:28:43,593 initializer.py:23] Successful!
    INFO 2022-06-06 09:28:43,593 initializer.py:24]
    INFO 2022-06-06 09:28:43,593 processor.py:97] Loading evaluating model ...
    INFO 2022-06-06 09:28:43,640 processor.py:102] Successful!
    INFO 2022-06-06 09:28:43,640 processor.py:103]
    INFO 2022-06-06 09:28:43,640 processor.py:106] Starting evaluating ...
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1183/1183 [02:11<00:00,  8.98it/s]
    INFO 2022-06-06 09:30:55,437 processor.py:78] Top-1 accuracy: 17929/18928(94.72%), Top-5 accuracy: 18795/18928(99.30%), Mean loss:0.1804
    INFO 2022-06-06 09:30:55,438 processor.py:81] Evaluating time: 131.80s, Speed: 143.62 sequnces/(second*GPU)
    INFO 2022-06-06 09:30:55,438 processor.py:84]
    INFO 2022-06-06 09:30:55,438 processor.py:108] Finish evaluating!



x-sub 预测结果:

    INFO 2022-06-06 08:48:27,548 initializer.py:23] Successful!
    INFO 2022-06-06 08:48:27,548 initializer.py:24]
    INFO 2022-06-06 08:48:27,548 processor.py:97] Loading evaluating model ...
    INFO 2022-06-06 08:48:27,600 processor.py:102] Successful!
    INFO 2022-06-06 08:48:27,600 processor.py:103]
    INFO 2022-06-06 08:48:27,600 processor.py:106] Starting evaluating ...
    100%|██████████████████████████████████████████████████████████████████████████████████████████| 1030/1030 [01:55<00:00,  8.93it/s]
    INFO 2022-06-06 08:50:22,966 processor.py:78] Top-1 accuracy: 14868/16480(90.22%), Top-5 accuracy: 16238/16480(98.53%), Mean loss:0.3653
    INFO 2022-06-06 08:50:22,966 processor.py:81] Evaluating time: 115.36s, Speed: 142.85 sequnces/(second*GPU)
    INFO 2022-06-06 08:50:22,966 processor.py:84]
    INFO 2022-06-06 08:50:22,966 processor.py:108] Finish evaluating!


## 5. 项目结构
    ├─data	
    ├─configs # 配置文件
    │  ├─2001.yaml
    │  ├─2002.yaml
    ├─dataset # 数据加载类
    │  ├─npy_dataset
    │  │  │─graph.py
    │  │  │─ntu_feeder.py
    ├─model # 模型结构
    │  │  │─activations.py 
    │  │  │─attentions.py
    │  │  │─initializer.py
    │  │  │─layers.py
    │  │  │─nets.py
    ├─reader # 数据预处理模块
    │  │  │─ignor.txt # 记录ntu中缺少骨架的部分
    │  │  │─ntu_reader.py
    │  │  │─transformer.py
    ├─scheduler # 学习率变化
    │  │  │─lr_schedulers.py 
    ├─generator.py
    ├─infer.py # 模型推理
    ├─initializer.py
    ├─main.py 
    ├─processor.py
    ├─utils.py
    │
    │
## 6. 模型动转静   
#### 6.1 模型动转静
以x-sub数据集作为推理示范
运行

    python main.py -c 2001 -ex -pp xsub.pdparams // 这里可以转换成自己的模型目录
执行完成后将在xsub.pdparams同级文件夹下生成pdmodel, pdiparams, 以及pdiparams.info 文件。
转换输出log如下

    INFO 2022-06-05 20:31:20,772 initializer.py:44] Saving folder path: ./workdir/temp
    INFO 2022-06-05 20:31:20,772 initializer.py:14]
    INFO 2022-06-05 20:31:20,772 initializer.py:15] Starting preparing ...
    INFO 2022-06-05 20:31:20,776 initializer.py:44] Saving folder path: ./workdir/temp
    INFO 2022-06-05 20:31:20,776 initializer.py:39] Saving model name: xsub.pdparams
    INFO 2022-06-05 20:31:20,791 initializer.py:70] Dataset: ntu-xsub
    INFO 2022-06-05 20:31:20,791 initializer.py:71] Batch size: train-16, eval-16
    INFO 2022-06-05 20:31:20,791 initializer.py:72] Data shape (branch, channel, frame, joint, person): [3, 6, 288, 25, 2]
    INFO 2022-06-05 20:31:20,791 initializer.py:73] Number of action classes: 60
    W0605 20:31:22.415406 1489696 gpu_context.cc:278] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 11.4, Runtime API Version: 11.2
    W0605 20:31:22.419064 1489696 gpu_context.cc:306] device: 0, cuDNN Version: 8.4.
    INFO 2022-06-05 20:39:19,794 initializer.py:83] Model: EfficientGCN-B0 {'stem_channel': 64, 'block_args': [[48, 1, 0.5], [24, 1, 0.5], [64, 2, 1], [128, 2, 1]], 'fusion_stage': 2, 'act_type': 'swish', 'att_type': 'stja', 'layer_type': 'SG', 'drop_prob': 0.25, 'kernel_size': [5, 2], 'scale_args': [1.2, 1.35], 'expand_ratio': 0, 'reduct_ratio': 2, 'bias': True, 'edge': True}
    INFO 2022-06-05 20:39:19,796 initializer.py:107] LR_Scheduler: cosine {'max_epoch': 70, 'warm_up': 10}
    INFO 2022-06-05 20:39:19,797 initializer.py:99] Optimizer: Momentum {'learning_rate': <paddle.optimizer.lr.LambdaDecay object at 0x7f694ffe1ee0>, 'momentum': 0.9, 'use_nesterov': True, 'weight_decay': 0.0001}
    INFO 2022-06-05 20:39:19,797 initializer.py:110] Loss function: CrossEntropyLoss
    INFO 2022-06-05 20:39:19,797 initializer.py:23] Successful!
    INFO 2022-06-05 20:39:19,797 initializer.py:24]
    INFO 2022-06-05 20:39:19,797 processor.py:150] Loading model ...
    INFO 2022-06-05 20:39:22,976 processor.py:159] Successful!
    INFO 2022-06-05 20:39:22,976 processor.py:160]
  
转换后的模型可在[static model](https://github.com/Wuxiao85/paddle_EfficientGCNv/blob/main/pretrain_model/) 获取


    
#### 6.2 模型推理

## 7. TIPC
## 参考及引用
    @article{song2022constructing,
      author    = {Song, Yi-Fan and Zhang, Zhang and Shan, Caifeng and Wang, Liang},
      title     = {Constructing stronger and faster baselines for skeleton-based action recognition},
      journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year      = {2022},
      publisher = {IEEE},
      url       = {https://doi.org/10.1109/TPAMI.2022.3157033},
      doi       = {10.1109/TPAMI.2022.3157033}
    }
    
    @inproceedings{song2020stronger,
      author    = {Song, Yi-Fan and Zhang, Zhang and Shan, Caifeng and Wang, Liang},
      title     = {Stronger, Faster and More Explainable: A Graph Convolutional Baseline for Skeleton-Based Action Recognition},
      booktitle = {Proceedings of the 28th ACM International Conference on Multimedia (ACMMM)},
      pages     = {1625--1633},
      year      = {2020},
      isbn      = {9781450379885},
      publisher = {Association for Computing Machinery},
      address   = {New York, NY, USA},
      url       = {https://doi.org/10.1145/3394171.3413802},
      doi       = {10.1145/3394171.3413802},
    }
* [PaddlePaddle](https://github.com/paddlepaddle/paddle)

感谢飞浆提供的算力和支持。

