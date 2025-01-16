import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x

class GatedDeConv2dWithActivation(torch.nn.Module):
    """
    Gated DeConvlution layer with activation (default activation:LeakyReLU)
    resize + conv
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 bias=True, batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        # print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)

class DBGatedConv(nn.Sequential):
    '''
    double 3x3 conv layers with Batch normalization and ReLU
    '''

    def __init__(self, in_channels, out_channels):
        conv_layers = [
            GatedConv2dWithActivation(in_channels, out_channels, 3),
            GatedConv2dWithActivation(out_channels, out_channels, 3),
        ]
        super(DBGatedConv, self).__init__(*conv_layers)

class GatedContractingPath(nn.Module):
    def __init__(self, in_channels, first_outchannels):
        super(GatedContractingPath, self).__init__()
        self.conv1 = DBGatedConv(in_channels, first_outchannels)
        self.conv2 = DBGatedConv(first_outchannels, first_outchannels * 2)
        self.conv3 = DBGatedConv(first_outchannels * 2, first_outchannels * 4)
        self.conv4 = DBGatedConv(first_outchannels * 4, first_outchannels * 8)

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        output1 = self.conv1(x)  # (N, 64, 568, 568)
        output = self.maxpool(output1)  # (N, 64, 284, 284)
        output2 = self.conv2(output)  # (N, 128, 280, 280)
        output = self.maxpool(output2)  # (N, 128, 140, 140)
        output3 = self.conv3(output)  # (N, 256, 136, 136)
        output = self.maxpool(output3)  # (N, 256, 68, 68)
        output4 = self.conv4(output)  # (N, 512, 64, 64)
        output = self.maxpool(output4)  # (N, 512, 32, 32)
        return output1, output2, output3, output4, output

class GatedExpansivePath(nn.Module):
    '''
    pass1, pass2, pass3, pass4 are the featuremaps passed from Contracting path
    '''

    def __init__(self, in_channels):
        super(GatedExpansivePath, self).__init__()
        # input (N, 1024, 28, 28)
        self.upconv1 = GatedDeConv2dWithActivation(2, in_channels, in_channels // 2, 3, 1)  # (N, 512, 56, 56)
        self.conv1 = DBGatedConv(in_channels, in_channels // 2)  # (N, 512, 52, 52)

        self.upconv2 = GatedDeConv2dWithActivation(2, in_channels // 2, in_channels // 4, 3, 1)  # (N, 256, 104, 104)
        self.conv2 = DBGatedConv(in_channels // 2, in_channels // 4)  # (N, 256, 100, 100)

        self.upconv3 = GatedDeConv2dWithActivation(2, in_channels // 4, in_channels // 8, 3, 1)  # (N, 128, 200, 200)
        self.conv3 = DBGatedConv(in_channels // 4, in_channels // 8)  # (N, 128, 196, 196)

        self.upconv4 = GatedDeConv2dWithActivation(2, in_channels // 8, in_channels // 16, 3, 1)  # (N, 64, 392, 392)
        self.conv4 = DBGatedConv(in_channels // 8, in_channels // 16)  # (N, 64, 388, 388)
        #self.conv5 = GatedConv2dWithActivation( in_channels // 16, in_channels // 32, 3)
        # for match output shape with

    def forward(self, x, pass1, pass2, pass3, pass4):#
        output = self.upconv1(x)
        output = torch.cat((output, pass4), 1)
        output = self.conv1(output)

        output = self.upconv2(output)
        output = torch.cat((output, pass3), 1)
        output = self.conv2(output)

        output = self.upconv3(output)
        output = torch.cat((output, pass2), 1)
        output = self.conv3(output)

        output = self.upconv4(output)
        output = torch.cat((output, pass1), 1)
        output = self.conv4(output)
        return output

class DilatedGatedConv(nn.Sequential):
    '''
    double 3x3 conv layers with Batch normalization and ReLU
    '''

    def __init__(self, in_channels, out_channels):
        conv_layers = [
            GatedConv2dWithActivation(in_channels, in_channels, 3, padding=2, dilation=2),
            GatedConv2dWithActivation(in_channels, in_channels, 3, padding=4, dilation=4),
            GatedConv2dWithActivation(in_channels, in_channels, 3, padding=8, dilation=8),
            GatedConv2dWithActivation(in_channels, in_channels, 3, padding=16, dilation=16),
            GatedConv2dWithActivation(in_channels, out_channels, 3),
        ]
        super(DilatedGatedConv, self).__init__(*conv_layers)
class GatedUnet(nn.Module):
    def __init__(self, in_channels=3, first_outchannels=64):
        super(GatedUnet, self).__init__()
        self.contracting_path = GatedContractingPath(in_channels=in_channels, first_outchannels=first_outchannels)
        self.middle_conv = DilatedGatedConv(first_outchannels * 8, first_outchannels * 16)
        self.expansive_path = GatedExpansivePath(in_channels=first_outchannels * 16)
        self.conv_last = GatedConv2dWithActivation(first_outchannels, 3, 3, activation=None)
        self.tanh = nn.Tanh()

    def forward(self, x):
        pass1, pass2, pass3, pass4, output = self.contracting_path(x)
        output = self.middle_conv(output)
        output = self.expansive_path(output, pass1, pass2, pass3, pass4)
        output = self.conv_last(output)
        output = self.tanh(output)
        return output