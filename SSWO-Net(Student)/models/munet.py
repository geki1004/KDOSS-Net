import torch
import torch.nn as nn


class MobileNetV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(MobileNetV2Block, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
class MobileNetV2Block2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(MobileNetV2Block2, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = None
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.expand is not None:
            x = self.expand(x)
        depthwise_output = self.depthwise(x)
        out = self.project(depthwise_output)
        if self.use_residual:
            return x + out, depthwise_output
        else:
            return out, depthwise_output


class MobileUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MobileUNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.down_block(64, 128)
        self.enc3 = self.down_block(128, 256)
        self.enc4 = self.down_block(256, 512)

        # Bottleneck
        self.bottleneck = self.down_block(512, 1024)

        # Decoder
        self.up4 = self.up_block(1024, 512)
        self.dec4= self.conv_block(1024,512)
        self.up3 = self.up_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.up_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.up_block(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            MobileNetV2Block(in_channels, out_channels, stride=1),
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with size alignment
        d4 = self.dec4(torch.cat((self.up4(b),e4), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4),e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3),e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2),e1), dim=1))
        # Final output
        out = self.final_conv(d1)
        return out#,b,d1
class MUNet_S(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(MUNet_S, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.down_block(64, 128)
        self.enc3 = self.down_block(128, 256)
        self.enc4 = self.down_block(256, 512)

        # Bottleneck
        self.bottleneck = self.down_block_bottle(512, 1024)

        # Decoder
        self.up4 = self.up_block(1024, 512)
        self.dec4= self.conv_block(1024,512)
        self.up3 = self.up_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.up_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.up_block(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            MobileNetV2Block(in_channels, out_channels, stride=1),
        )

    def down_block_bottle(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            MobileNetV2Block2(in_channels, out_channels, stride=1),
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b, bb = self.bottleneck(e4)

        # Decoder with size alignment
        d4 = self.dec4(torch.cat((self.up4(b),e4), dim=1))
        d3 = self.dec3(torch.cat((self.up3(d4),e3), dim=1))
        d2 = self.dec2(torch.cat((self.up2(d3),e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2),e1), dim=1))
        # Final output
        out = self.final_conv(d1)
        return out,bb,d1
