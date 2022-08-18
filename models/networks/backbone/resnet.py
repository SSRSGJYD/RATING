import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.modules.batchnorm import _BatchNorm
from .layer_norm import norm2d
from .activation import activation

__all__ = ['resnet']

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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm2d(planes,**kwargs)
        self.activation = activation(**kwargs)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm2d(planes,**kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm2d(planes,**kwargs)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm2d(planes,**kwargs)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm2d(planes * self.expansion, **kwargs)
        self.activation = activation(**kwargs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet(nn.Module):

    def __init__(self, 
                block, 
                layers, 
                in_channels=3,
                num_classes=1000, 
                frozen_stages=-1,
                zero_init_residual=False,
                norm_eval=False,
                output_layers=1,
                **kwargs):
        super().__init__()

        self.inplanes = 64
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.output_layers = output_layers

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm2d(64, **kwargs)
        self.activation = activation(**kwargs)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm2d(planes * block.expansion, **kwargs)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        xs = []
        x = self.layer1(x)
        if self.output_layers >= 4:
            xs.append(self.avgpool(x))
        x = self.layer2(x)
        if self.output_layers >= 3:
            xs.append(self.avgpool(x))
        x = self.layer3(x)
        if self.output_layers >= 2:
            xs.append(self.avgpool(x))
        x = self.layer4(x)
        if self.output_layers >= 1:
            xs.append(self.avgpool(x))

        if len(xs) == 1:
            return xs[0]
        else:
            return xs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

def resnet(arch, in_channel=3, pretrained=False, output_layers=1, **kwargs):
    """Constructs a ResNet model. """

    assert (arch in ['10','18','34','50','101','152']), 'The depth is not right'
    arch_settings = {
        '10': (BasicBlock, [1, 1, 1, 1]),
        '18': (BasicBlock, [2, 2, 2, 2]),
        '34': (BasicBlock, [3, 4, 6, 3]),
        '50': (Bottleneck, [3, 4, 6, 3]),
        '101': (Bottleneck, [3, 4, 23, 3]),
        '152': (Bottleneck, [3, 8, 36, 3])
    }

    block, width = arch_settings[arch]
    basic_feature_dim = block.expansion * 512
    total_feature_dim = 0
    for i in range(output_layers):
        total_feature_dim += basic_feature_dim // (2 ** i)

    valid_kwargs = dict()
    for key, value in kwargs.items():
        if key in ['frozen_stages']:
            valid_kwargs[key] = value
    
    model = ResNet(block, width, in_channel, output_layers=output_layers, **valid_kwargs)
    if pretrained and arch in ['18','34','50','101','152']:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet'+arch]), strict=False)
    return model, total_feature_dim
    
    
