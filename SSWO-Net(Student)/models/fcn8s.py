import torch
import torch.nn as nn
import torchvision.models as models


class FCN(nn.Module):
    def __init__(self, num_classes, num_conv_filters=1024, use_bias=True, weight_decay=0.,init_weights=True):
        super(FCN, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True)

        # Remove fully connected layers and keep only convolutional layers
        self.features = self.vgg16.features

        # Add additional convolutional layers
        self.conv6 = nn.Conv2d(512, num_conv_filters, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.conv7 = nn.Conv2d(num_conv_filters, num_conv_filters, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.conv7.weight, mode='fan_out', nonlinearity='relu')

        self.score_fr = nn.Conv2d(num_conv_filters, num_classes, kernel_size=1)

        # Upsampling layers
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1,
                                                bias=False)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)

        if init_weights:
            print('initialize weights...')
            self._initialize_weights()

    def forward(self, x):
        # Encoder
        x3 = self.features[:17](x)
        x4 = self.features[17:24](x3)
        x5 = self.features[24:](x4)

        # FCN-8s
        score_fr = self.score_fr(self.drop7(self.relu7(self.conv7(self.drop6(self.relu6(self.conv6(x5)))))))
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(x4)
        fuse_pool4 = upscore2 + score_pool4
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3 = self.score_pool3(x3)
        fuse_pool3 = upscore_pool4 + score_pool3
        upscore8 = self.upscore8(fuse_pool3)

        return upscore8
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # He initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)