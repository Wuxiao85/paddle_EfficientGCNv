## 基于PaddlePaddle实现的 EfficientGCNv1
#### 简介
[EfficientGCN: Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition](https://paperswithcode.com/paper/constructing-stronger-and-faster-baselines)
一文提出了基于骨架行为识别的baseline，在论文中，将基于骨架识别的网络分为input branch和 main stream两部分。Input branch 用于提取骨架数据的多模态特征，提取的特征通过concat等操作完成特征融合后将输入main stream中预测动作分类。
![EfficientGCN](./images/model.PNG)
[官方源码](https://gitee.com/yfsong0709/EfficientGCNv1)
#### 数据集和复现精度
###### NTU-RGB-D60
下载地址: https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H
###### 复现精度（待调整）
|      | x-sub | x-view |
|:----:|:-----:|:------:| 
|  论文  | 90.2  |  94.9  | 
| 复现精度 | 89.9  | 94.66  |
#### 准备环境
#### 快速开始
###### 1. 下载数据集
到官网下载好数据集
###### 2. 下载本项目及训练权重
    git clone git@github.com:Wuxiao85/paddle_EfficientGCNv.git
预训练网络下载: [xsub.pdparams](https://github.com/Wuxiao85/paddle_EfficientGCNv/blob/main/pretrain_model/xsub.pdparams), [xview.pdparams](https://github.com/Wuxiao85/paddle_EfficientGCNv/blob/main/pretrain_model/xview.pdparams)
###### 3. 数据预处理
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
###### 4. 模型训练
执行

    # export CUDA_VISIBLE_DEVICES=0 // 用到的gpu编号
    # python main.py -c 2001 // 在x-sub数据集上训练

    # export CUDA_VISIBLE_DEVICES=0 // 用到的gpu编号
    # python main.py -c 2002 // 在x-view数据集上训练

训练模型将保留在temp文件夹下。
训练log: [xview.log](https://github.com/Wuxiao85/paddle_EfficientGCNv/blob/main/workdir/2002_EfficientGCN-B0_ntu-xview/2022-06-05%2000-51-10/log.txt), [xsub.log](https://github.com/Wuxiao85/paddle_EfficientGCNv/blob/main/workdir/2001_EfficientGCN-B0_ntu-xsub/2022-06-05%2000-52-01/log.txt)
部分训练log：
x-view：
    
    [ 2022-06-05 14:13:19,296 ] Epoch: 70/70, Training accuracy: 37307/37632(99.14%), Training time: 644.48s
    [ 2022-06-05 14:13:19,296 ] 
    [ 2022-06-05 14:13:19,297 ] Evaluating for epoch 70/70 ...
    [ 2022-06-05 14:15:30,463 ] Top-1 accuracy: 17868/18928(94.40%), Top-5 accuracy: 18775/18928(99.19%), Mean loss:0.2006
    [ 2022-06-05 14:15:30,464 ] Evaluating time: 131.16s, Speed: 144.31 sequnces/(second*GPU)
    [ 2022-06-05 14:15:30,464 ] 
    [ 2022-06-05 14:15:30,464 ] Saving model for epoch 70/70 ...
    [ 2022-06-05 14:15:30,491 ] Best top-1 accuracy: 94.66%, Total time: 00d-13h-19m-50s
    [ 2022-06-05 14:15:30,491 ] 
    [ 2022-06-05 14:15:30,491 ] Finish training!
    [ 2022-06-05 14:15:30,491 ] 
    
    [ 2022-06-05 14:15:30,491 ] Best top-1 accuracy: 94.66%, Total time: 00d-13h-19m-50s
    [ 2022-06-05 14:15:30,491 ] 
    [ 2022-06-05 14:15:30,491 ] Finish training!
    [ 2022-06-05 14:15:30,491 ] 
x-sub:
    
    [ 2022-06-05 15:07:25,503 ] Evaluating for epoch 70/70 ...
    [ 2022-06-05 15:09:20,578 ] Top-1 accuracy: 14815/16480(89.90%), Top-5 accuracy: 16218/16480(98.41%), Mean loss:0.3894
    [ 2022-06-05 15:09:20,578 ] Evaluating time: 115.07s, Speed: 143.21 sequnces/(second*GPU)
    [ 2022-06-05 15:09:20,578 ] 
    [ 2022-06-05 15:09:20,578 ] Saving model for epoch 70/70 ...
    [ 2022-06-05 15:09:20,605 ] Best top-1 accuracy: 89.90%, Total time: 00d-14h-14m-20s
    [ 2022-06-05 15:09:20,605 ] 
    [ 2022-06-05 15:09:20,605 ] Finish training!
    [ 2022-06-05 15:09:20,605 ] 
    [ 2022-06-05 15:09:20,605 ] Best top-1 accuracy: 89.90%, Total time: 00d-14h-14m-20s
    [ 2022-06-05 15:09:20,605 ] 
    [ 2022-06-05 15:09:20,605 ] Finish training!
    [ 2022-06-05 15:09:20,605 ] 
###### 5. 模型预测
执行

    # export CUDA_VISIBLE_DEVICES=0 // 用到的gpu编号
    # python main.py -c 2001 -e -pp <path to pretrain>// 在x-sub数据集上训练

    # export CUDA_VISIBLE_DEVICES=0 // 用到的gpu编号
    # python main.py -c 2002 // 在x-view数据集上训练

x-view 预测结果:
    
    INFO 2022-06-05 14:44:24,114 initializer.py:24]
    INFO 2022-06-05 14:44:24,114 processor.py:97] Loading evaluating model ...
    INFO 2022-06-05 14:44:24,159 processor.py:102] Successful!
    INFO 2022-06-05 14:44:24,159 processor.py:103]
    INFO 2022-06-05 14:44:24,160 processor.py:107] Starting evaluating ...
    100%|█████████████████████████████████████████████████████████████████████████████████| 1183/1183 [02:46<00:00,  7.12it/s]
    INFO 2022-06-05 14:47:10,345 processor.py:78] Top-1 accuracy: 17918/18928(94.66%), Top-5 accuracy: 18789/18928(99.27%), Mean loss:0.1837
    INFO 2022-06-05 14:47:10,345 processor.py:81] Evaluating time: 166.18s, Speed: 113.90 sequnces/(second*GPU)
    INFO 2022-06-05 14:47:10,346 processor.py:84]
    INFO 2022-06-05 14:47:10,346 processor.py:109] Finish evaluating!

x-sub 预测结果:

    INFO 2022-06-05 15:46:22,605 processor.py:97] Loading evaluating model ...
    INFO 2022-06-05 15:46:22,649 processor.py:102] Successful!
    INFO 2022-06-05 15:46:22,649 processor.py:103]
    INFO 2022-06-05 15:46:22,649 processor.py:106] Starting evaluating ...
    100%|█████████████████████████████████████████████████████████████████████████████████| 1030/1030 [01:56<00:00,  8.86it/s]
    INFO 2022-06-05 15:48:18,917 processor.py:78] Top-1 accuracy: 14814/16480(89.89%), Top-5 accuracy: 16218/16480(98.41%), Mean loss:0.3897
    INFO 2022-06-05 15:48:18,918 processor.py:81] Evaluating time: 116.27s, Speed: 141.74 sequnces/(second*GPU)
    INFO 2022-06-05 15:48:18,918 processor.py:84]
    INFO 2022-06-05 15:48:18,918 processor.py:108] Finish evaluating!

#### 项目结构
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
    ├─transport_model.py
    ├─utils.py
    │
    │
#### 模型动转静   
###### 1. 模型动转静
###### 2. 模型推理
#### TIPC
#### 参考及引用
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

