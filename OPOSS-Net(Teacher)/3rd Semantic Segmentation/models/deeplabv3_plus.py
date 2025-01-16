from typing import Dict ,List
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

class BasicConv(nn.Sequential):
    """
    Basic Conv Block : conv2d - batchnorm - act
    """
    def __init__(self, in_channels:int, out_channels:int, kernel_size, stride:int=1, padding:int=0, dilation=1, groups=1, bias=True, norm='batch', act:nn.Module=nn.ReLU()):
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
        if norm=='instance':
            modules += [nn.InstanceNorm2d(out_channels)]
        if norm=='batch':
            modules += [nn.BatchNorm2d(out_channels)]
        if act is not None:
            modules += [act]
        super().__init__(*modules)

### ResNet backbone
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet50(nn.Module):
    def __init__(self,  num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, pretrained=True):
        super(ResNet50, self).__init__()
        ## resnet50
        layers=[3, 4, 6, 3]
        block=Bottleneck
        ###
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(6, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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
        if pretrained:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth',
                                              progress=True)
            self.load_state_dict(state_dict)
            
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

### intermediate layer feature getter from resnet50
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers:Dict):
        """Module wrapper that returns intermediate layers from a model

        Args:
            model : model on which we will extract the features
            return_layers (Dict[name, return_name]): 
                a dictionary containing the names of modules for which the activations will be returned as the key of the dict, 
                and value of the dict is the name of the returned activation
        """
        # return layer로 지정된 이름의 레이어가 model 내에 없으면 에러 발생
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        
        self.return_layers = copy.deepcopy(return_layers)       
        # ResNet class로 생성되어있는 모델 객체를 모듈이름:모듈클래스로 이뤄진 OrderedDict()로 변형
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            # return_layer를 모두 포함할 때까지 반복
            if name in return_layers:
                del return_layers[name]
            if not return_layers: break
        
        super(IntermediateLayerGetter, self).__init__(layers)
    def forward(self, x):
        output = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers: # return_layers key에 name이 있으면
                output[self.return_layers[name]] = x # output name으로 지정한 이름으로 현재 레이어의 출력을 저장
        return output
    
#### ASPP module, deeplab head
class ASPPConv(nn.Sequential):
    ''' Atrous Convolution Layer in ASPP module '''
    def __init__(self, in_channels: int, out_channels: int, dilation: int, kernel_size=3, act=nn.ReLU()):
        modules = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation, bias=False), # inputsize == outputsize
            nn.BatchNorm2d(out_channels),
        ]
        if act is not None:
            assert isinstance(act, nn.Module), "act must be nn.Module instance"
            modules += [act]
        super().__init__(*modules)

class ASPPPooling(nn.Sequential):
    ''' pooling layer in ASPP module in DeepLap v3 '''
    def __init__(self, in_channels: int, out_channels: int, act=nn.ReLU()):
        super().__init__(
            nn.AdaptiveAvgPool2d(1), # GAP
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            act
        )
        
    def forward(self, x: torch.Tensor):
        size = x.shape[-2:] # (H, W)
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates:List[int], out_channels:int=256 ):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 conv
        modules += [nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())]
        for rate in atrous_rates:
            modules += [ASPPConv(in_channels, out_channels, rate)]

        modules += [ASPPPooling(in_channels, out_channels)]
        self.convlist = nn.ModuleList(modules)
        self.project = nn.Sequential(BasicConv(len(self.convlist) * (out_channels), out_channels, 1, norm='batch', act=nn.ReLU()), nn.Dropout(0.1)) 
        
    def forward(self, x):
        conv_results = []
        for conv in self.convlist:
            conv_results += [conv(x)]
        output = torch.cat(conv_results, dim=1)
        output = self.project(output)
        return output

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate):
        super().__init__()
        self.project = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
#### ResNet50 backbone + deeplabv3 plus
class Resnet50_DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes=3, output_stride=8, pretrained=False):
        super().__init__()
        
        if output_stride==8:
            replace_stride_with_dilation=[False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation=[False, False, True]
            aspp_dilate = [6, 12, 18]
        
        backbone = ResNet50(replace_stride_with_dilation=replace_stride_with_dilation, pretrained=pretrained)
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        in_channels = 2048
        low_level_channels = 256
        classifier = DeepLabHeadV3Plus(in_channels, low_level_channels, num_classes, aspp_dilate)
        
        self.backbone = backbone
        self.classifier = classifier
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x