import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
import gdown
from .quantization import *

model_urls = {
    'resnet50': 'https://github.com/iamsh4shank/BFA_ViT/blob/main/cifar-10_models/cifar_resnet50.pt',
}


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = quan_Conv2d(inplanes,
                                planes,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = quan_Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = quan_Conv2d(3,
                                 64,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = quan_Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class CifarResNet50(nn.Module):
    def __init__(self, out_classes):
      super(CifarResNet50, self).__init__()
      self.arch = ResNet(Bottleneck, [3, 4, 6, 3])
      self.fc1 = quan_Linear(1000, 500)
      self.bn1 = nn.BatchNorm1d(500)
      self.fc2 = quan_Linear(500, 200)
      self.bn2 = nn.BatchNorm1d(200)
      self.fc3 = quan_Linear(200, 100)
      self.bn3 = nn.BatchNorm1d(100)
      self.fc4 = quan_Linear(100, out_classes)
      self.relu = nn.ReLU()

    def forward(self, image):
      output = self.arch(image)
      output = self.relu(self.bn1(self.fc1(output)))
      output = self.relu(self.bn2(self.fc2(output)))
      output = self.relu(self.bn3(self.fc3(output)))
      output = self.relu(self.fc4(output))
      return output


class CifarResNet(nn.Module):
    """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
    def __init__(self, block, depth, num_classes):
        """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
        super(CifarResNet, self).__init__()

        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth -
                2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print('CifarResNet : Depth : {} , Layers for each block : {}'.format(
            depth, layer_blocks))

        self.num_classes = num_classes

        self.conv_1_3x3 = quan_Conv2d(3,
                                    16,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = quan_Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion,
                                     stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def resnet20_quan(num_classes=10):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 20, num_classes)
    return model


def resnet32_quan(num_classes=10):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 32, num_classes)
    return model


def resnet44_quan(num_classes=10):
    """Constructs a ResNet-44 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 44, num_classes)
    return model


def resnet56_quan(num_classes=10):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 56, num_classes)
    return model


def resnet110_quan(num_classes=10):
    """Constructs a ResNet-110 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = CifarResNet(ResNetBasicblock, 110, num_classes)
    return model

def resnet50_quan(num_classes=10, pretrained = True):
    model = CifarResNet50(num_classes)

    if(pretrained):

      model.load_state_dict(model_zoo.load_url(model_urls['resnet50'])
      
    return model