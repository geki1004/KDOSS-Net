import torch
import torch.nn as nn
import torchvision

class SegNet(nn.Module):
    def __init__(self, in_channels=3, feature_channels=512, num_classes=3, init_weights=False):
        super(SegNet, self).__init__()
        self.encoder = Encoder(in_channels, feature_channels, mode='train')
        self.decoder = Decoder(feature_channels, num_classes)

        if init_weights:
            print('initialize weights...')
            self._initialize_weights()

    def forward(self, x):
        enc_out, indices = self.encoder(x)
        out = self.decoder(enc_out, indices)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # He initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



class Encoder(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, mode:str='train') :
        super().__init__()
        assert mode in ['train', 'test'], 'mode must be train or test'
        make_block_config = [(out_channels//8, 2), (out_channels//4, 2), (out_channels//2, 3), (out_channels, 3), (out_channels, 3)]
        l = []
        for channel, num_blocks in make_block_config:
            l += [self._make_basic_block(in_channels, channel, num_blocks)]
            in_channels=channel
        self.conv_block_1 = l[0]
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv_block_2 = l[1]
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv_block_3 = l[2]
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv_block_4 = l[3]
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.conv_block_5 = l[4]
        self.maxpool_5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)


        if mode=='train':
            self.load_pretrained_vgg16()

    def forward(self, x):
        out = self.conv_block_1(x)
        out, ind_1 = self.maxpool_1(out)

        out = self.conv_block_2(out)
        out, ind_2 = self.maxpool_2(out)

        out = self.conv_block_3(out)
        out, ind_3 = self.maxpool_3(out)

        out = self.conv_block_4(out)
        out, ind_4 = self.maxpool_4(out)

        out = self.conv_block_5(out)
        out, ind_5 = self.maxpool_5(out)
        return out, [ind_1, ind_2, ind_3, ind_4, ind_5]

    def _make_basic_block(self, in_channels, out_channels, num_blocks):
        if num_blocks==2:
            l = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(),
                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(),
                 ]

        if num_blocks==3:
            l = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(),
                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(),
                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(),
                 ]
        return nn.Sequential(*l)

    def load_pretrained_vgg16(self):
        print('load pretrained vgg16...')
        pt_vgg16 = torchvision.models.vgg16_bn(pretrained=True).features #weights='IMAGENET1K_V1'
        for mod, prt in zip(self.state_dict(), pt_vgg16.state_dict()):
            # print(mod, prt)
            self.state_dict()[mod].copy_(pt_vgg16.state_dict()[prt])
            # print(self.state_dict()[mod]==pt_vgg16.state_dict()[prt])

class Decoder(nn.Module):
    def __init__(self, in_channels:int, out_channels:int) :
        super().__init__()
        make_block_config = [(in_channels, 3), (in_channels//2, 3), (in_channels//4, 3), (in_channels//8, 2), (out_channels, 2)]
        l = []
        for channel, num_blocks in make_block_config:
            l += [self._make_basic_block(in_channels, channel, num_blocks)]
            in_channels=channel

        self.unpool_1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_block_1 = l[0]

        self.unpool_2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_block_2 = l[1]

        self.unpool_3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_block_3 = l[2]

        self.unpool_4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_block_4 = l[3]

        self.unpool_5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_block_5 = l[4]

    def forward(self, x, indices):
        out = self.unpool_1(x, indices[-1])
        out = self.conv_block_1(out)

        out = self.unpool_2(out, indices[-2])
        out = self.conv_block_2(out)

        out = self.unpool_3(out, indices[-3])
        out = self.conv_block_3(out)

        out = self.unpool_4(out, indices[-4])
        out = self.conv_block_4(out)

        out = self.unpool_5(out, indices[-5])
        out = self.conv_block_5(out)

        return out

    def _make_basic_block(self, in_channels, out_channels, num_blocks):
        if num_blocks==2:
            l = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(in_channels),
                 nn.ReLU(),
                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(),
                 ]

        if num_blocks==3:
            l = [nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(in_channels),
                 nn.ReLU(),
                 nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(in_channels),
                 nn.ReLU(),
                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(),
                 ]
        return nn.Sequential(*l)



