"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch.nn as nn
from torch.utils import model_zoo
import kornia
import torch
import torch.nn.functional as F
from dropblock import DropBlock2D


__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']

pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
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

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, in_ch=3, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(in_ch, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(in_ch, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        print(self.layer0.conv1)

        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model_bak(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def initialize_pretrained_model(model, num_classes, settings, in_ch):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    state_dict = model_zoo.load_url(settings['url'])

    if in_ch == 1:
        conv1_weight = state_dict['layer0.conv1.weight']
        state_dict['layer0.conv1.weight'] = conv1_weight.mean(dim=1, keepdim=True)
    model.load_state_dict(state_dict)


    #model.load_state_dict()
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def senet154(num_classes=1000, pretrained='imagenet', in_ch=3):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16, in_ch=in_ch,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings, in_ch=in_ch)
    return model


def se_resnet50(num_classes=1000, pretrained='imagenet', in_ch=3):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False, in_ch=in_ch,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings, in_ch=in_ch)
    return model


def se_resnet101(num_classes=1000, pretrained='imagenet', in_ch=3):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False, in_ch=in_ch,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, num_classes, settings, in_ch=in_ch)
    return model


def se_resnet152(num_classes=1000, pretrained='imagenet', in_ch=3):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False, in_ch=in_ch,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings, in_ch=in_ch)
    return model


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet', in_ch=3):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False, in_ch=in_ch,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings, in_ch=in_ch)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet', in_ch=3):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False, in_ch=in_ch,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings, in_ch=in_ch)
    return model


from torch.nn.parameter import Parameter
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


import torch
import torch.nn as nn
import torch.nn.functional as F  #(uncomment if needed,but you likely already have it)

#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch / FastAI by lessw2020
#github: https://github.com/lessw2020/mish

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))



class se_resnext50_32x4d_bengali(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.layer0 = base.layer0
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4#
        # Final linear layer
        self._dropout = nn.Dropout(0.4)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._max_pooling = nn.AdaptiveMaxPool2d(1)
        #self._max_blur_pool = kornia.contrib.MaxBlurPool2d(kernel_size=3, ceil_mode=True)
        out_channels = 2048




        # # ##################liner0###################
        # #vowel_diacritic
        # self.fc1 = nn.Linear(out_channels, 11)
        # # grapheme_root
        # self.fc2 = nn.Linear(out_channels, 168)
        # # consonant_diacritic
        # self.fc3 = nn.Linear(out_channels, 7)




        ###################liner1###################
        self.liner = nn.Linear(out_channels, 186)



        # ###################liner2###################
        # self.liner1 = nn.Linear(2048, 512)
        # self.act = nn.LeakyReLU()
        # self.liner2 = nn.Linear(512, 186)




        # ###################liner3###################
        # # vowel_diacritic
        # self.act = nn.LeakyReLU()
        # self.fc1 = nn.Linear(out_channels, 512)
        # self.fc11 = nn.Linear(512, 11)
        # # grapheme_root
        # self.fc2 = nn.Linear(out_channels, 512)
        # self.fc22 = nn.Linear(512, 168)
        # # consonant_diacritic
        # self.fc3 = nn.Linear(out_channels, 512)
        # self.fc33 = nn.Linear(512, 7)





        # ###################liner4###################
        # ##Gem
        # self.gem = GeM()
        # self.relu_tail = nn.ReLU()
        # self.bnorm = nn.BatchNorm2d(512)
        # self.conv1 = nn.Conv2d(out_channels, 512, kernel_size=3, stride=1, padding=1)
        # self.line1 = nn.Linear(512, 11)
        #
        # self.conv2 = nn.Conv2d(out_channels, 512, kernel_size=3, stride=1, padding=1)
        # self.line2 = nn.Linear(512, 168)
        #
        # self.conv3 = nn.Conv2d(out_channels, 512, kernel_size=3, stride=1, padding=1)
        # self.line3 = nn.Linear(512, 7)



        # ###################liner4 GEM###################
        ##Gem
        # self.gem = GeM()
        # self.relu_tail = nn.ReLU()
        # self.bnorm = nn.BatchNorm2d(512)
        # self.conv1 = nn.Conv2d(out_channels, 512, kernel_size=3, stride=1, padding=1)
        # self.line1 = nn.Linear(512, 11)
        #
        # self.conv2 = nn.Conv2d(out_channels, 512, kernel_size=3, stride=1, padding=1)
        # self.line2 = nn.Linear(512, 168)
        #
        # self.conv3 = nn.Conv2d(out_channels, 512, kernel_size=3, stride=1, padding=1)
        # self.line3 = nn.Linear(512, 7)



        # ##################liner5###################
        # #vowel_diacritic
        # self.fc11 = nn.Linear(out_channels, 512)
        # self.fc12 = nn.Linear(512, 11)
        # # grapheme_root
        # self.fc2 = nn.Linear(out_channels, 168)
        # # consonant_diacritic
        # self.fc3 = nn.Linear(out_channels, 7)
        # self.act = nn.ELU()


        # ##################liner6###################
        # self.drop_block0 = DropBlock2D(block_size=16, drop_prob=0.2)
        # self.drop_block1 = DropBlock2D(block_size=8, drop_prob=0.2)
        # #vowel_diacritic
        # self.fc1 = nn.Linear(out_channels, 11)
        # # grapheme_root
        # self.fc2 = nn.Linear(out_channels, 168)
        # # consonant_diacritic
        # self.fc3 = nn.Linear(out_channels, 7)



    def forward(self, inputs):
        # ###################base features###################

        bs = inputs.size(0)
        # convert to 3 channel
        #inputs = inputs.repeat(1,3,1,1)

        x = self.layer0(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print('fdsafsa:', x.shape)




        # ###################liner0###################
        # x_avg = self._avg_pooling(x)
        # x_max = self._max_pooling(x)
        # x = 0.5 * (x_avg + x_max)
        # x = x.view(bs, -1)
        # x = self._dropout(x)
        # x1 = self.fc1(x)
        # x2 = self.fc2(x)
        # x3 = self.fc3(x)
        # return x1, x2, x3


        #################liner1################### with cutmix and RandomShiftRotate 30 augment gives CV 0.9913
        x_avg = self._avg_pooling(x)
        x_max = self._max_pooling(x)
        x = 0.5 * (x_avg + x_max)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self.liner(x)
        preds = torch.split(x, [11, 168, 7], dim=1)
        x1 = preds[0]
        x2 = preds[1]
        x3 = preds[2]
        return x1, x2, x3




        # ###################liner2###################
        # x_avg = self._avg_pooling(x)
        # x_max = self._max_pooling(x)
        # x = 0.5 * (x_avg + x_max)
        # x = x.view(bs, -1)
        # x = self._dropout(x)
        # x = self.act(self.liner1(x))
        # x = self.liner2(x)
        # preds = torch.split(x, [11, 168, 7], dim=1)
        # x1 = preds[0]
        # x2 = preds[1]
        # x3 = preds[2]
        # return x1, x2, x3




        # ###################liner3###################
        # x_avg = self._avg_pooling(x)
        # x_max = self._max_pooling(x)
        # x = 0.5 * (x_avg + x_max)
        # x = x.view(bs, -1)
        # x1 = self._dropout(self.act(self.fc1(x)))
        # x1 = self.fc11(x1)
        #
        # x2 = self._dropout(self.act(self.fc2(x)))
        # x2 = self.fc22(x2)
        #
        # x3 = self._dropout(self.act(self.fc3(x)))
        # x3 = self.fc33(x3)
        # return x1, x2, x3




        # # ###################liner4###################
        # #print('x0:', x.shape)
        # x1 = self.relu_tail(x)
        # x1 = self.conv1(x1)
        # #print('x1:', x1.shape)
        # x1 = self.bnorm(x1)
        # x1 = self._avg_pooling(x1)
        # #print('x2:', x1.shape)
        # x1 = x1.view(bs, -1)
        # x1 = self.line1(x1)
        #
        #
        #
        # x2 = self.relu_tail(x)
        # x2 = self.conv1(x2)
        # x2 = self.bnorm(x2)
        # x2 = self._avg_pooling(x2)
        # x2 = x2.view(bs, -1)
        # x2 = self.line2(x2)
        #
        #
        # x3 = self.relu_tail(x)
        # x3 = self.conv1(x3)
        # x3 = self.bnorm(x3)
        # x3 = self._avg_pooling(x3)
        # x3 = x3.view(bs, -1)
        # x3 = self.line3(x3)
        # return x1, x2, x3







        # ###################liner4 GEM###################
        # #print('x0:', x.shape)
        # x1 = self.relu_tail(x)
        # x1 = self.conv1(x1)
        # #print('x1:', x1.shape)
        # x1 = self.bnorm(x1)
        # x1 = self.gem(x1)
        # #print('x2:', x1.shape)
        # x1 = x1.view(bs, -1)
        # x1 = self.line1(x1)
        #
        #
        #
        # x2 = self.relu_tail(x)
        # x2 = self.conv1(x2)
        # x2 = self.bnorm(x2)
        # x2 = self.gem(x2)
        # x2 = x2.view(bs, -1)
        # x2 = self.line2(x2)
        #
        #
        # x3 = self.relu_tail(x)
        # x3 = self.conv1(x3)
        # x3 = self.bnorm(x3)
        # x3 = self.gem(x3)
        # x3 = x3.view(bs, -1)
        # x3 = self.line3(x3)
        # return x1, x2, x3


        # ###################liner5###################
        # x_avg = self._avg_pooling(x)
        # x_max = self._max_pooling(x)
        # x = 0.5 * (x_avg + x_max)
        # x = x.view(bs, -1)
        #
        # x1 = self._dropout(self.act(self.fc11(x)))
        # x1 = self.fc12(x1)
        #
        # x2 = self.fc2(x)
        # x3 = self.fc3(x)
        # return x1, x2, x3



        # # ###################liner 6 base features###################
        # bs = inputs.size(0)
        # # convert to 3 channel
        # inputs = inputs.repeat(1,3,1,1)
        # x = self.layer0(inputs)
        # x = self.drop_block0(x)
        # x = self.layer1(x)
        # x = self.drop_block1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # #print('fdsafsa:', x.shape)
        #
        #
        # ###################liner0###################
        # x_avg = self._avg_pooling(x)
        # x_max = self._max_pooling(x)
        # x = 0.5 * (x_avg + x_max)
        # x = x.view(bs, -1)
        # x = self._dropout(x)
        # x1 = self.fc1(x)
        # x2 = self.fc2(x)
        # x3 = self.fc3(x)
        # return x1, x2, x3





def se50_32_4d_resnext(pretrained='imagenet', in_ch=1):
    base = se_resnext50_32x4d(pretrained=pretrained, in_ch=in_ch)
    net = se_resnext50_32x4d_bengali(base=base)
    return net


if __name__ == '__main__':
    from torchstat import stat
    import torch
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    path = '/data1/wangwenpeng/Bengali/seresnext50_0.5_newaug_liner0/global_max_recall.pth'

    base = se_resnext50_32x4d(pretrained='imagenet', in_ch=3)
    net = se_resnext50_32x4d_bengali(base=base)

    #net = nn.DataParallel(net).cuda()
    #net.load_state_dict(torch.load(path))
    print(net)

    img = torch.rand([1, 1, 68, 118])
    #img = img.cuda()

    x1, x2, x3 = net(img)
    print(x1.shape, x2.shape, x3.shape)

    #stat(net, (1, 224, 224))









