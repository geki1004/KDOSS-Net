import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels:int, out_channels:int, kernel_size, stride:int=1, padding:int=0, dilation:int=1, groups=1, bias:bool=False, bn:bool=True, act:nn.Module=nn.PReLU()):
        l = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)]
        if bn:
            l += [nn.BatchNorm2d(out_channels)]
        if act != None:
            l += [act]
        super().__init__(*l)
        

class InputInjection(nn.Sequential):
    def __init__(self, num_pool):
        l = [nn.AvgPool2d(3, stride=2, padding=1)] * num_pool
        super().__init__(*l)

        
class CGBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, dilation:int, reduction:int=16) :
        super().__init__()
        # 1x1 conv
        self.conv_1x1 = ConvBlock(in_channels, out_channels//2, 1, bias=False, bn=True, act=nn.PReLU())
        # depthwise 3x3 
        self.f_loc = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=1, groups=out_channels//2, bias=False) 
        self.f_sur = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=dilation, dilation=dilation, groups=out_channels//2, bias=False) 
        # BN, PReLU after concatenation
        self.bn_prelu = nn.Sequential(nn.BatchNorm2d(out_channels),
                                      nn.PReLU())
        # global context
        self.fglo_avgpool = nn.AdaptiveMaxPool2d(1)
        self.fglo_linear = nn.Sequential(
                                    nn.Linear(out_channels, out_channels//reduction),
                                    nn.ReLU(),
                                    nn.Linear(out_channels//reduction, out_channels),
                                    nn.Sigmoid()
                                   )
    def forward(self, x):
        out = self.conv_1x1(x)
        loc_out = self.f_loc(out)
        sur_out = self.f_sur(out)
        joint = torch.cat([loc_out, sur_out], dim=1)
        joint = self.bn_prelu(joint)
        
        B, C, _, _ = joint.shape
        glo_out = self.fglo_avgpool(joint).view(B, C)
        glo_out = self.fglo_linear(glo_out).view(B, C, 1, 1)
        output = joint * glo_out
        return output + x # residual connection ( GRL )

class CGBlock_Down(nn.Module):
    """ CGBlock downsampling version. H->H//2, W->W//2 """
    def __init__(self, in_channels:int, out_channels:int, dilation:int, reduction:int=16) :
        super().__init__()
        # 1x1 conv
        self.downconv_3x3 = ConvBlock(in_channels, out_channels, 3, stride=2, padding=1, bias=False, bn=True, act=nn.PReLU()) # H->H/2, W->W/2
        # depthwise 3x3 
        self.f_loc = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False) 
        self.f_sur = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, groups=out_channels, bias=False) 
        
        # BN, PReLU after concatenation
        self.bn_prelu = nn.Sequential(nn.BatchNorm2d(2*out_channels),
                                      nn.PReLU())
        
        ## channel reduction conv
        self.ch_reduction = nn.Conv2d(2*out_channels, out_channels, kernel_size=1, bias=False) 
        
        # global context
        self.fglo_avgpool = nn.AdaptiveMaxPool2d(1)
        self.fglo_linear = nn.Sequential(
                                    nn.Linear(out_channels, out_channels//reduction),
                                    nn.ReLU(),
                                    nn.Linear(out_channels//reduction, out_channels),
                                    nn.Sigmoid()
                                   )
    def forward(self, x):
        out = self.downconv_3x3(x)
        loc_out = self.f_loc(out)
        sur_out = self.f_sur(out)
        joint = torch.cat([loc_out, sur_out], dim=1)
        joint = self.bn_prelu(joint)
        joint = self.ch_reduction(joint)
        
        B, C, _, _ = joint.shape
        glo_out = self.fglo_avgpool(joint).view(B, C)
        glo_out = self.fglo_linear(glo_out).view(B, C, 1, 1)
        output = joint * glo_out
        return output # no residual 
    
class CGNet(nn.Module):
    def __init__(self, in_channels:int, num_classes:int, M=3, N=21) :
        """Context Guided Network(CGNet)

        Args:
            in_channels (int): the number of channels of input image
            num_classes (int): the number of classes
            M (int, optional): the number of CGBlocks in stage 2. Defaults to 3.
            N (int, optional): the number of CGBlocks in stage 3. Defaults to 21.
        """
        super().__init__()
        # Additional input
        self.inp_inj_1 = InputInjection(1) # down 1/2
        self.inp_inj_2 = InputInjection(2) # down 1/4
        
        # stage 1
        self.stage_1 = nn.Sequential(ConvBlock(in_channels, 32, 3, stride=2, padding=1, bias=False, bn=True, act=nn.PReLU()),
                                     ConvBlock(32, 32, 3, stride=1, padding=1, bias=False, bn=True, act=nn.PReLU()),
                                     ConvBlock(32, 32, 3, stride=1, padding=1, bias=False, bn=True, act=nn.PReLU()))
        
        # stage 2
        self.bn_prelu_1 = nn.Sequential(nn.BatchNorm2d(32+in_channels), nn.PReLU())
        
        self.down_cg_st2 = CGBlock_Down(32+in_channels, 64, dilation=2, reduction=8)
        self.stage_2 = nn.Sequential(*[CGBlock(64, 64, dilation=2, reduction=8)]*M)
        
        #s stage 3
        self.bn_prelu_2 = nn.Sequential(nn.BatchNorm2d(128+in_channels), nn.PReLU())
        
        self.down_cg_st3 = CGBlock_Down(128+in_channels, 128, dilation=4, reduction=16)
        self.stage_3 = nn.Sequential(*[CGBlock(128, 128, dilation=4, reduction=16)]*N)
        
        self.bn_prelu_3 = nn.Sequential(nn.BatchNorm2d(256), nn.PReLU())
        
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)
        
        
    def forward(self, x):
        # Additional input
        inp_half = self.inp_inj_1(x)
        inp_qua = self.inp_inj_2(x)
        
        # stage 1
        output1 = self.stage_1(x)
        # stage 2 input
        output1= self.bn_prelu_1(torch.cat([output1, inp_half], dim=1))
        # stage 2
        output2_0 = self.down_cg_st2(output1)
        output2 = self.stage_2(output2_0)
        # stage 3 input
        output2 = self.bn_prelu_2(torch.cat([output2, output2_0, inp_qua], dim=1))
        # stage 3
        output3_0 = self.down_cg_st3(output2)
        output3 = self.stage_3(output3_0)
        
        output3 = self.bn_prelu_3(torch.cat([output3_0, output3], dim=1))
        
        output = self.classifier(output3)
        output = F.interpolate(output, x.shape[-2:], mode='bilinear')
        return output
        