import torch
import torch.nn as nn
import torch.nn.functional as F

class DBConv(nn.Sequential):
    '''
    double 3x3 conv layers with Batch normalization and ReLU
    '''

    def __init__(self, in_channels, out_channels):
        conv_layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(DBConv, self).__init__(*conv_layers)

class ContractingPath(nn.Module):
    def __init__(self, in_channels, first_outchannels):
        super(ContractingPath, self).__init__()
        self.conv1 = DBConv(in_channels, first_outchannels)
        self.conv2 = DBConv(first_outchannels, first_outchannels * 2)
        self.conv3 = DBConv(first_outchannels * 2, first_outchannels * 4)
        self.conv4 = DBConv(first_outchannels * 4, first_outchannels * 8)

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


class ExpansivePath(nn.Module):
    '''
    pass1, pass2, pass3, pass4 are the featuremaps passed from Contracting path
    '''

    def __init__(self, in_channels):
        super(ExpansivePath, self).__init__()
        # input (N, 1024, 28, 28)
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2)  # (N, 512, 56, 56)
        self.conv1 = DBConv(in_channels, in_channels // 2)  # (N, 512, 52, 52)
        self.upconv2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 2, 2)  # (N, 256, 104, 104)
        self.conv2 = DBConv(in_channels // 2, in_channels // 4)  # (N, 256, 100, 100)
        self.upconv3 = nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 2, 2)  # (N, 128, 200, 200)
        self.conv3 = DBConv(in_channels // 4, in_channels // 8)  # (N, 128, 196, 196)
        self.upconv4 = nn.ConvTranspose2d(in_channels // 8, in_channels // 16, 2, 2)  # (N, 64, 392, 392)
        self.conv4 = DBConv(in_channels // 8, in_channels // 16)  # (N, 64, 388, 388)

        # for match output shape with

    def forward(self, x, pass1, pass2, pass3, pass4):
        # input (N, 1024, 28, 28)
        output = self.upconv1(x)  # (N, 512, 56, 56)
        output = torch.cat((output, pass4), 1)  # (N, 1024, 64, 64)
        output = self.conv1(output)  # (N, 512, 60, 60)

        output = self.upconv2(output)  # (N, 256, 120, 120)
        output = torch.cat((output, pass3), 1)  # (N, 512, 136, 136)
        output = self.conv2(output)  # (N, 256, 132, 132)

        output = self.upconv3(output)  # (N, 128, 264, 264)
        output = torch.cat((output, pass2), 1)  # (N, 256, 280, 280)
        output = self.conv3(output)  # (N, 128, 276, 276)

        output = self.upconv4(output)  # (N, 64, 552, 552)
        output = torch.cat((output, pass1), 1)  # (N, 128, 568, 568)
        output = self.conv4(output)  # (N, 64, 564, 564)

        return output



class Unet(nn.Module):
    def __init__(self, in_channels=3, first_outchannels=64, num_classes=3):
        super(Unet, self).__init__()
        self.contracting_path = ContractingPath(in_channels=in_channels, first_outchannels=first_outchannels)
        self.middle_conv = DBConv(first_outchannels * 8, first_outchannels * 16)
        self.expansive_path = ExpansivePath(in_channels=first_outchannels * 16)
        self.conv_1x1 = nn.Conv2d(first_outchannels, num_classes, 1)

    def forward(self, x):
        pass1, pass2, pass3, pass4, output = self.contracting_path(x)
        mid = self.middle_conv(output)
        last = self.expansive_path(mid, pass1, pass2, pass3, pass4)
        output = self.conv_1x1(last)

        return output, mid, last