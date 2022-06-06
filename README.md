# 基于PaddlePaddle实现的 EfficientGCNv1
## 1. 简介
[EfficientGCN: Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition](https://paperswithcode.com/paper/constructing-stronger-and-faster-baselines)
一文提出了基于骨架行为识别的baseline，在论文中，将基于骨架识别的网络分为input branch和 main stream两部分。Input branch 用于提取骨架数据的多模态特征，提取的特征通过concat等操作完成特征融合后将输入main stream中预测动作分类。

![EfficientGCN](./images/model.PNG)

[官方源码](https://gitee.com/yfsong0709/EfficientGCNv1)
## 2. 数据集和复现精度
#### 2.1 NTU-RGB-D60
下载地址: https://drive.google.com/open?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H
#### 2.2 复现精度
|   EfficientGCN-B0   | x-sub | x-view |
|:----:|:-----:|:------:| 
|  论文  | 90.2  |  94.9  | 
| 复现精度 | 90.22  | 94.82  |
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

    [ 2022-06-06 11:19:57,983 ] Epoch: 68/70, Training accuracy: 37334/37632(99.21%), Training time: 648.86s
    [ 2022-06-06 11:19:57,983 ] 
    [ 2022-06-06 11:19:57,984 ] Evaluating for epoch 68/70 ...
    [ 2022-06-06 11:22:10,828 ] Top-1 accuracy: 17948/18928(94.82%), Top-5 accuracy: 18785/18928(99.24%), Mean loss:0.1831
    [ 2022-06-06 11:22:10,828 ] Evaluating time: 132.84s, Speed: 142.49 sequnces/(second*GPU)
    [ 2022-06-06 11:22:10,828 ] 
    [ 2022-06-06 11:22:10,828 ] Saving model for epoch 68/70 ...
    [ 2022-06-06 11:22:10,855 ] Best top-1 accuracy: 94.82%, Total time: 00d-01h-30m-37s
    [ 2022-06-06 11:22:10,856 ] 
    [ 2022-06-06 11:33:02,045 ] Epoch: 69/70, Training accuracy: 37312/37632(99.15%), Training time: 651.19s
    [ 2022-06-06 11:33:02,046 ] 
    [ 2022-06-06 11:33:02,046 ] Evaluating for epoch 69/70 ...
    [ 2022-06-06 11:35:13,177 ] Top-1 accuracy: 17933/18928(94.74%), Top-5 accuracy: 18790/18928(99.27%), Mean loss:0.1836
    [ 2022-06-06 11:35:13,178 ] Evaluating time: 131.13s, Speed: 144.35 sequnces/(second*GPU)
    [ 2022-06-06 11:35:13,178 ] 
    [ 2022-06-06 11:35:13,178 ] Saving model for epoch 69/70 ...
    [ 2022-06-06 11:35:13,205 ] Best top-1 accuracy: 94.82%, Total time: 00d-01h-43m-40s
    [ 2022-06-06 11:35:13,205 ] 
    [ 2022-06-06 11:46:00,135 ] Epoch: 70/70, Training accuracy: 37335/37632(99.21%), Training time: 646.93s
    [ 2022-06-06 11:46:00,136 ] 
    [ 2022-06-06 11:46:00,136 ] Evaluating for epoch 70/70 ...
    [ 2022-06-06 11:48:11,561 ] Top-1 accuracy: 17874/18928(94.43%), Top-5 accuracy: 18785/18928(99.24%), Mean loss:0.1973
    [ 2022-06-06 11:48:11,561 ] Evaluating time: 131.42s, Speed: 144.02 sequnces/(second*GPU)
    [ 2022-06-06 11:48:11,561 ] 
    [ 2022-06-06 11:48:11,561 ] Saving model for epoch 70/70 ...
    [ 2022-06-06 11:48:11,588 ] Best top-1 accuracy: 94.82%, Total time: 00d-01h-56m-38s
    [ 2022-06-06 11:48:11,589 ] 
    [ 2022-06-06 11:48:11,589 ] Finish training!
    [ 2022-06-06 11:48:11,589 ] 
 

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
    
    INFO 2022-06-06 12:22:45,003 initializer.py:23] Successful!
    INFO 2022-06-06 12:22:45,003 initializer.py:24]
    INFO 2022-06-06 12:22:45,003 processor.py:97] Loading evaluating model ...
    INFO 2022-06-06 12:22:45,048 processor.py:102] Successful!
    INFO 2022-06-06 12:22:45,048 processor.py:103]
    INFO 2022-06-06 12:22:45,048 processor.py:106] Starting evaluating ...
    100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1183/1183 [02:12<00:00,  8.96it/s]
    INFO 2022-06-06 12:24:57,131 processor.py:78] Top-1 accuracy: 17948/18928(94.82%), Top-5 accuracy: 18785/18928(99.24%), Mean loss:0.1830
    INFO 2022-06-06 12:24:57,131 processor.py:81] Evaluating time: 132.08s, Speed: 143.31 sequnces/(second*GPU)
    INFO 2022-06-06 12:24:57,131 processor.py:84]
    INFO 2022-06-06 12:24:57,131 processor.py:108] Finish evaluating!




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
**生成tiny数据**
在生成tiny数据集之前，需要先按数据预处理的步骤将数据先划分成xview和xsub.

    python dataset/tiny_data_gen.py --data_path <path to data geneorated before> --label_path <path to label file> --data_num <default 5*16> --save_dir <path to tiny data>

**静态模型推理**
运行

    python infer.py --model_file <path to pdmodel> --params_file <path to pdiparams> --save_dir <directory to save tiny dataset> --b <batch_size>

静态推理时用到的batchsize需要与 config 文件中 test_batch_size 一致。

以下用xsub数据集上训练的模型示范静态推理过程。

    python infer.py --model_file xsub.pdmodel --params_file xsub.pdimodel --save_dir ./dataset/ntu --b <batch_size>

输出结果如下：

    Batch action class Predict:  [52 53 54 55 55 57 58 59  0  1  2  3  4  5  6  7] Batch action class True:  [52, 53, 54, 55, 56, 57, 58, 59, 0, 1, 2, 3, 4, 5, 6, 7] Batch Accuracy:  0.9375 Batch sample Name:  ['S004C001P007R002A053.skeleton', 'S004C001P007R002A054.skeleton', 'S004C001P007R002A055.skeleton', 'S004C001P007R002A056.skeleton', 'S004C001P007R002A057.skeleton', 'S004C001P007R002A058.skeleton', 'S004C001P007R002A059.skeleton', 'S004C001P007R002A060.skeleton', 'S004C001P020R001A001.skeleton', 'S004C001P020R001A002.skeleton', 'S004C001P020R001A003.skeleton', 'S004C001P020R001A004.skeleton', 'S004C001P020R001A005.skeleton', 'S004C001P020R001A006.skeleton', 'S004C001P020R001A007.skeleton', 'S004C001P020R001A008.skeleton']
    Batch action class Predict:  [ 8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23] Batch action class True:  [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] Batch Accuracy:  1.0 Batch sample Name:  ['S004C001P020R001A009.skeleton', 'S004C001P020R001A010.skeleton', 'S004C001P020R001A011.skeleton', 'S004C001P020R001A012.skeleton', 'S004C001P020R001A013.skeleton', 'S004C001P020R001A014.skeleton', 'S004C001P020R001A015.skeleton', 'S004C001P020R001A016.skeleton', 'S004C001P020R001A017.skeleton', 'S004C001P020R001A018.skeleton', 'S004C001P020R001A019.skeleton', 'S004C001P020R001A020.skeleton', 'S004C001P020R001A021.skeleton', 'S004C001P020R001A022.skeleton', 'S004C001P020R001A023.skeleton', 'S004C001P020R001A024.skeleton']
    Batch action class Predict:  [24 25 26 27 28 29 30 28 32 33 34 35 36 37 38 39] Batch action class True:  [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] Batch Accuracy:  0.9375 Batch sample Name:  ['S004C001P020R001A025.skeleton', 'S004C001P020R001A026.skeleton', 'S004C001P020R001A027.skeleton', 'S004C001P020R001A028.skeleton', 'S004C001P020R001A029.skeleton', 'S004C001P020R001A030.skeleton', 'S004C001P020R001A031.skeleton', 'S004C001P020R001A032.skeleton', 'S004C001P020R001A033.skeleton', 'S004C001P020R001A034.skeleton', 'S004C001P020R001A035.skeleton', 'S004C001P020R001A036.skeleton', 'S004C001P020R001A037.skeleton', 'S004C001P020R001A038.skeleton', 'S004C001P020R001A039.skeleton', 'S004C001P020R001A040.skeleton']
    Batch action class Predict:  [40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55] Batch action class True:  [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55] Batch Accuracy:  1.0 Batch sample Name:  ['S004C001P020R001A041.skeleton', 'S004C001P020R001A042.skeleton', 'S004C001P020R001A043.skeleton', 'S004C001P020R001A044.skeleton', 'S004C001P020R001A045.skeleton', 'S004C001P020R001A046.skeleton', 'S004C001P020R001A047.skeleton', 'S004C001P020R001A048.skeleton', 'S004C001P020R001A049.skeleton', 'S004C001P020R001A050.skeleton', 'S004C001P020R001A051.skeleton', 'S004C001P020R001A052.skeleton', 'S004C001P020R001A053.skeleton', 'S004C001P020R001A054.skeleton', 'S004C001P020R001A055.skeleton', 'S004C001P020R001A056.skeleton']
    Batch action class Predict:  [56 57 58 59  0 27  2  3 23  5 30  7  8 38 10 11] Batch action class True:  [56, 57, 58, 59, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] Batch Accuracy:  0.75 Batch sample Name:  ['S004C001P020R001A057.skeleton', 'S004C001P020R001A058.skeleton', 'S004C001P020R001A059.skeleton', 'S004C001P020R001A060.skeleton', 'S004C001P020R002A001.skeleton', 'S004C001P020R002A002.skeleton', 'S004C001P020R002A003.skeleton', 'S004C001P020R002A004.skeleton', 'S004C001P020R002A005.skeleton', 'S004C001P020R002A006.skeleton', 'S004C001P020R002A007.skeleton', 'S004C001P020R002A008.skeleton', 'S004C001P020R002A009.skeleton', 'S004C001P020R002A010.skeleton', 'S004C001P020R002A011.skeleton', 'S004C001P020R002A012.skeleton']
    Infer Mean Accuracy:  0.925


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

