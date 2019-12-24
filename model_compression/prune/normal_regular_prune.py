import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
# import util
from models import nin
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', action='store', default='../data', help='dataset path')
parser.add_argument('--cpu', action='store_true', help='disables CUDA training')
# percent(剪枝率)
parser.add_argument('--percent', type=float, default=0.5, help='nin:0.5')
# 正常|规整剪枝标志
parser.add_argument('--normal_regular', type=int, default=1, help='--normal_regular_flag (default: normal)')
# model层数
parser.add_argument('--layers', type=int, default=9, help='layers (default: 9)')
# 稀疏训练后的model
parser.add_argument('--model', default='models_save/nin_preprune.pth', type=str, metavar='PATH', help='path to raw trained model (default: none)')
# 剪枝后保存的model
parser.add_argument('--save', default='models_save/nin_prune.pth', type=str, metavar='PATH', help='path to save prune model (default: none)')
args = parser.parse_args()
base_number = args.normal_regular
layers = args.layers
print(args)

if base_number <= 0:
    print('\r\n!base_number is error!\r\n')
    base_number = 1

# 定义模型， 并导入参数!
model = nin.Net()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        model.load_state_dict(torch.load(args.model)['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
print('旧模型: ', model)
# ===================================================================

total = 0 # 所有BN层的channel之和
i = 0   # i 为 batchnorm 的统计层数
for m in model.modules(): 
    if isinstance(m, nn.BatchNorm2d): 
        if i < layers - 1:
            i += 1
            total += m.weight.data.shape[0] # m.weight.data.shape[0] 为每个BN层的通道数!

bn = torch.zeros(total) # 一个 list， 每个通道一个值
index = 0
i = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1
            size = m.weight.data.shape[0] # bn层的通道数
            bn[index:(index+size)] = m.weight.data.abs().clone() # 将所有bn层的参数都放在bn中
            index += size

y, j = torch.sort(bn) # 排序  y 是排序后的值， j 是排序的index
thre_index = int(total * args.percent) # 确定阈值， 这里指留下多少个值
if thre_index == total:  # 至少要留下一个
    thre_index = total - 1
thre_0 = y[thre_index] # thre_0， 找到对应的阈值

#********************************预剪枝*********************************
pruned = 0
cfg_0 = [] # 存储每一层的通道数
cfg = [] # 存储每一层留下的通道数!
cfg_mask = [] # 存储每一层的 mask
i = 0

for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if i < layers - 1:
            i += 1

            weight_copy = m.weight.data.clone() # 获取这行的权重值
            mask = weight_copy.abs().gt(thre_0).float() # 获取大于阈值的index
            remain_channels = torch.sum(mask) # 只是为了统计剩下的channels

            if remain_channels == 0:  # 处理一下剪枝剪没了的情况
                print('\r\n!please turn down the prune_ratio!\r\n')
                remain_channels = 1
                mask[int(torch.argmax(weight_copy))]=1

            # ****************** 规整剪枝 ****************** # 对剪枝之后的网络进行规整!
            v = 0
            n = 1
            if remain_channels % base_number != 0: # 对剩余通道进行处理 base number 默认等于1， 也就是对所有通道都不进行处理!
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                if remain_channels > base_number:
                    while v < remain_channels:
                        n += 1
                        v = base_number * n
                    if remain_channels - (v - base_number) < v - remain_channels:
                        remain_channels = v - base_number
                    else:
                        remain_channels = v
                    if remain_channels > m.weight.data.size()[0]:
                        remain_channels = m.weight.data.size()[0]
                    remain_channels = torch.tensor(remain_channels)
                        
                    y, j = torch.sort(weight_copy.abs())
                    thre_1 = y[-remain_channels]
                    mask = weight_copy.abs().ge(thre_1).float()
            pruned = pruned + mask.shape[0] - torch.sum(mask) # 剪枝通道数
            m.weight.data.mul_(mask) # 剪枝， 将对应的位置置为0
            m.bias.data.mul_(mask) # 剪枝， 将对应的位置置为0
            cfg_0.append(mask.shape[0]) 
            cfg.append(int(remain_channels))
            cfg_mask.append(mask.clone())

            print('layer_index: {:d} \t total_channel: {:d} \t remaining_channel: {:d} \t pruned_ratio: {:f}'.
                format(k, mask.shape[0], int(torch.sum(mask)), (mask.shape[0] - torch.sum(mask)) / mask.shape[0])) # 打印log
pruned_ratio = float(pruned/total) # 计算预剪枝比率
print('\r\n!预剪枝完成!')
print('total_pruned_ratio: ', pruned_ratio)
#********************************预剪枝后model测试*********************************
def test():
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root = args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size = 64, shuffle=False, num_workers=1)
    model.eval()
    correct = 0
    
    for data, target in test_loader:
        if not args.cpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Accuracy: {:.2f}%\n'.format(acc))
    return
print('************预剪枝模型测试************')
if not args.cpu:
    model.cuda()
test()

#********************************剪枝*********************************
newmodel = nin.Net(cfg)   # 经过上一次剪枝之后的层
if not args.cpu:
    newmodel.cuda()
layer_id_in_cfg = 0

start_mask = torch.ones(3) # [1., 1., 1.] 初始通道数!
end_mask = cfg_mask[layer_id_in_cfg] # 取出第一层的预剪枝之后的层

i = 0
for [m0, m1] in zip(model.modules(), newmodel.modules()): 
    # m0 原始的层， m1 是预剪枝之后的层
    if isinstance(m0, nn.BatchNorm2d):  # 对 bn 层进行剪枝
        if i < layers - 1:  # 不是最后一层
            i += 1
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print("xxxx", idx1)
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            m1.weight.data = m0.weight.data[idx1].clone() # 将 weight, bias, running_mean, running_var 这几个复制到 m1 中
            m1.bias.data = m0.bias.data[idx1].clone() #
            m1.running_mean = m0.running_mean[idx1].clone() #
            m1.running_var = m0.running_var[idx1].clone() #

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  
                end_mask = cfg_mask[layer_id_in_cfg] # 取出下一层

        else:  # 最后一层就不要轻易动了
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

    elif isinstance(m0, nn.Conv2d): # 对卷积层进行剪枝
        if i < layers - 1:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w = m0.weight.data[:, idx0, :, :].clone()
            m1.weight.data = w[idx1, :, :, :].clone() # 复制指定的weight!
            m1.bias.data = m0.bias.data[idx1].clone()
        else:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0, :, :].clone()
            m1.bias.data = m0.bias.data.clone()

    elif isinstance(m0, nn.Linear):  # 对全连接层进行剪枝
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone() # 把全连接对应的层删除了

#******************************剪枝后model测试*********************************
print('新模型: ', newmodel)
print('**********剪枝后新模型测试*********')
model = newmodel
test()
#******************************剪枝后model保存*********************************
print('**********剪枝后新模型保存*********')
torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, args.save)
print('**********保存成功*********\r\n')

#*****************************剪枝前后model对比********************************
print('************旧模型结构************')
print(cfg_0)
print('************新模型结构************')
print(cfg, '\r\n')