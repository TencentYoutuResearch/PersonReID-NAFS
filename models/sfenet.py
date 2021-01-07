import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SfeNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3]):
        super(SfeNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.inplanes = 512
        self.branch2_layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.branch2_layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.inplanes = 1024
        self.branch3_layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, p2, p3):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)


        # branch 1
        b13 = self.layer3(x)
        b1 = self.layer4(b13)
        b1 = self.avgpool(b1)

        # branch 2
        b2 = self.height_shuffle(x, p2)
        b2 = self.branch2_layer3(b2)
        b2 = self.branch2_layer4(b2)
        b2 = self.recover_shuffle(b2, p2)

        index_pair_list_b2 = self.get_index_pair_list(b2, p2)
        part_feature_list_b2 = [self.avgpool(b2[:, :, pair[0]:pair[1], :]).squeeze() for pair in index_pair_list_b2]

        # branch 3
        b3 = self.height_shuffle(b13, p3)
        b3 = self.branch3_layer4(b3)
        b3 = self.recover_shuffle(b3, p3)

        index_pair_list_b3 = self.get_index_pair_list(b3, p3)
        part_feature_list_b3 = [self.avgpool(b3[:, :, pair[0]:pair[1], :]).squeeze() for pair in index_pair_list_b3]
        
        # #x = x.view(x.size(0), -1)
        # #x = self.fc(x)
        #
        # return x, feature_map_v

        return b1, part_feature_list_b2, part_feature_list_b3


    def get_index_pair_list(self, x, permu):
        batchsize, num_channels, height, width = x.data.size()
        number_slice = len(permu)
        height_per_slice = height // number_slice
        index_pair_list = [(height_per_slice*i, height_per_slice*(i+1)) for i in range(number_slice-1)]
        index_pair_list.append((height_per_slice*(number_slice-1), height))
        return index_pair_list


    def height_shuffle(self, x, permu):
        batchsize, num_channels, height, width = x.data.size()
        result = torch.zeros(batchsize, num_channels, height, width).cuda()
        number_slice = len(permu)
        height_per_slice = height // number_slice
        index_pair_list = [(height_per_slice*i, height_per_slice*(i+1)) for i in range(number_slice-1)]
        index_pair_list.append((height_per_slice*(number_slice-1), height))
        index_pair_list_shuffled = []
        for i in range(number_slice):
            index_pair_list_shuffled.append(index_pair_list[permu[i]])
        
        start = 0
        for i in range(len(index_pair_list_shuffled)):
            index_pair = index_pair_list_shuffled[i]
            length = index_pair[1] - index_pair[0]
            result[:, :, start:(start+length), :] = x[:, :, index_pair[0]:index_pair[1], :]
            start = start + length
        return result

    def recover_shuffle(self, x, permu):
        dic = {}
        recover_permu = []
        for i in range(len(permu)):
            dic[permu[i]] = i
        all_key = list(dic.keys())
        all_key.sort()
        for i in range(len(all_key)):
            recover_permu.append(dic[all_key[i]])

        return self.height_shuffle(x, recover_permu)
        
