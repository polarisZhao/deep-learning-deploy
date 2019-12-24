# deep-learning-deploy
deploy machine learning models into production

### 1. Deep Learning Framework

##### (1) Pytorch

**libtorch**

##### (2) Mxnet

##### (3) Tensorflow:tf-lite

### 2. deep learning mobile framework

##### (1) MNN

##### (2) ncnn

##### (3) coreml

##### (4) paddle-mobile

##### (5) MACE

### 3. Server

###### TensorRT

### 4. system

###### TVM

###### TC

### 5. Web

# 常见的模型压缩与加速方案

#### 蒸馏

Train a ResNet-18 model with knowledge distilled from a pre-trained ResNext-29 teacher

```
python3 train.py --model_dir experiments/resnet18_distill/resnext_teacher
```

+ Test accuracy: **94.788%**

#### 量化　(二值网络)



### 低秩分解

### 减枝

### References



蒸馏参考论文 

Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550.

剪枝 [Networks Slimming-Learning Efficient Convolutional Networks through Network Slimming](http://xxx.itp.ac.cn/pdf/1709.00513.pdf)

### 