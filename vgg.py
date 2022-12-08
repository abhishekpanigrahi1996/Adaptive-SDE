'''
Modified from https://github.com/pytorch/vision.git
'''

'''
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
#from torch.nn.common_types import _size_2_t



__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]



#currently works only for width = 1 or width = 2
class Conv2d(torch.nn.Conv2d) :
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        width: int = 1
    ) :
        super(Conv2d, self).__init__(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    dilation = dilation,
                    groups = groups,
                    bias = bias,
                    padding_mode = padding_mode
        )
        self.width = width
        
        if width == 2:      
            torch.manual_seed(0) 
            init_scale = 2./( in_channels * (kernel_size ** 2) )
            init_scale = init_scale ** (.25)

            A = torch.randn_like(self.weight) * init_scale
            B = torch.randn_like(self.weight) * init_scale
            self.a = torch.nn.Parameter(A, requires_grad=True) 
            self.b = torch.nn.Parameter(B, requires_grad=True) 
            
    def forward(self, input: Tensor) -> Tensor:     
        if self.width == 1:
            try:
                return self._conv_forward(input, self.weight, self.bias) 
            except:
                return self.forward(input) #This is necessary for Node 105
        else:
            return self._conv_forward(input, self.a * self.b, self.bias) #This becomes an issue in Node 105

    
class VGG(nn.Module):

    def __init__(self, features, width=1, dropout=0):
        super(VGG, self).__init__()
        self.features = features
        if dropout == 1:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, 10),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 10),
            )
        # Initialize weights
        for m in self.modules():
            
            #for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if width == 1:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, width=1):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2d(in_channels, v, kernel_size=3, padding=1, width=width)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(width=1, dropout=0):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], width=width), width=width, dropout=dropout)


def vgg11_bn(width=1, dropout=0):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True, width=width), width=width, dropout=dropout)


def vgg13(width=1, dropout=0):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B'], width=width), width=width, dropout=dropout)


def vgg13_bn(width=1, dropout=0):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True, width=width), width=width, dropout=dropout)


def vgg16(width=1, dropout=0):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'], width=width), width=width, dropout=dropout)


def vgg16_bn(width=1, dropout=0):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True, width=width), width=width, dropout=dropout)


def vgg19(width=1, dropout=0):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], width=width), width=width, dropout=dropout)


def vgg19_bn(width=1, dropout=0):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True, width=width), width=width, dropout=dropout)
'''
            
            
'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''


import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_scale=1, train_final_layer=1):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self.init_scale = init_scale
        self.train_final_layer = train_final_layer
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                print (self.init_scale)
                m.weight.data.normal_(0, 0.01 * self.init_scale)
                m.bias.data.zero_()
                
                if self.train_final_layer == 0:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


def make_layers(cfg, batch_norm=False, activation='relu'):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if activation == 'relu':
                act = nn.ReLU(inplace=True)
            elif activation == 'gelu':
                act = nn.GELU()
                
                
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            
            in_channels = v
    
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'AV': [64, 'N', 128, 'N', 256, 256, 'N', 512, 512, 'N', 512, 512, 'N'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'BV': [64, 64, 'N', 128, 128, 'N', 256, 256, 'N', 512, 512, 'N', 512, 512, 'N'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'DV': [64, 64, 'N', 128, 128, 'N', 256, 256, 256, 'N', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'EV': [64, 64, 'N', 128, 128, 'N', 256, 256, 256, 256, 'N', 512, 512, 512, 512, 'N', 512, 512, 512, 512, 'N'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model    


def vgg_manual(configuration='A', batch_norm=True, activation='relu', **kwargs):
    model = VGG(make_layers(cfg[configuration], batch_norm=batch_norm, activation=activation), **kwargs)
    return model    


