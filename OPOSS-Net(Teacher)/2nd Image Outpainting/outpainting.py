from functools import reduce
import torch.autograd as autograd
from torch.nn import Parameter
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import torch
import torch.nn.functional as F
import os
from html4vision import Col, imagetable
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models, utils
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.ops import masks_to_boxes
from metric import Measurement
import argparse
from tensorboardX import SummaryWriter
from datetime import datetime
import torchvision
from skimage import metrics
import skimage
from models import Unet, GatedUnet

seed_value = 123
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed_value)
np.random.seed(seed_value)

def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)
class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='replicate',
                 activation='none', norm='none', sn=False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=4, base_channels=64):
        super(PatchDiscriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: [B, 3, H, W]
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),  # Downsample
            nn.LeakyReLU(0.2, inplace=True),

            # Final output
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=0),  # Scalar output
        )

    def forward(self, img, mask):
        x = torch.cat((img, mask), 1)
        return self.net(x).view(x.size(0), -1)
def weights_init_normal(m, init_type='kaiming', gain=0.02):
    from torch.nn import init
    classname = m.__class__.__name__

    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)
def generate_html(G_net, D_net, device, data_loaders, html_save_path, max_rows=64):
    '''
    Visualizes one batch from both the training and validation sets.
    Images are stored in the specified HTML file path.
    '''
    G_net.eval()
    D_net.eval()
    torch.set_grad_enabled(False)
    if os.path.exists(html_save_path):
        shutil.rmtree(html_save_path)
    os.makedirs(html_save_path + '/images')

    # Evaluate examples
    for phase in ['train', 'val']:
        imgs, masked_imgs, seg_mask, b_mask, ob, name = next(iter(data_loaders[phase]))
        imgs = imgs.to(device)
        masked_imgs = masked_imgs.to(device)
        b_mask = b_mask.to(device)
        ob = ob.to(device)
        ob = ob * (1-b_mask)
        concat_input = torch.cat(((masked_imgs*2-1), (1-b_mask), ob), dim=1)
        outputs = G_net(concat_input)
        outputs = (outputs+1)/2
        outputs = imgs * b_mask + outputs * (1-b_mask)
        outputs = outputs.cpu()
        # Store images
        for i in range(min(imgs.shape[0], max_rows)):
            save_image(masked_imgs[i], html_save_path + '/images/' + phase + '_' + str(i) + '_masked.png')
            save_image(outputs[i], html_save_path + '/images/' + phase + '_' + str(i) + '_result.png')
            save_image(imgs[i], html_save_path + '/images/' + phase + '_' + str(i) + '_truth.png')

    # Generate table
    cols = [
        Col('id1', 'ID'),
        Col('img', 'Training set masked', html_save_path + '/images/train_*_masked.png'),
        Col('img', 'Training set result', html_save_path + '/images/train_*_result.png'),
        Col('img', 'Training set truth', html_save_path + '/images/train_*_truth.png'),
        Col('img', 'Validation set masked', html_save_path + '/images/val_*_masked.png'),
        Col('img', 'Validation set result', html_save_path + '/images/val_*_result.png'),
        Col('img', 'Validation set truth', html_save_path + '/images/val_*_truth.png'),
    ]
    imagetable(cols, out_file=html_save_path + '/index.html',
               pathrep=(html_save_path + '/images', 'images'))
    print('Generated image table at: ' + html_save_path + '/index.html')

def gram_matrix(feature_maps):
    (batch_size, num_channels, height, width) = feature_maps.size()
    features = feature_maps.view(batch_size, num_channels, height * width)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (num_channels * height * width)

class VGG19FeatLayer(nn.Module):
    def __init__(self, device=0):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval().cuda(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(device)

    def forward(self, x):
        out = {}
        x = x - self.mean
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        # print([x for x in out])
        return out

class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer, device=0, shallow_feats=False):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer(device=device)
        if shallow_feats:
            self.feat_style_layers = {'relu2_2': 1.0, 'relu3_2': 1.0}
            self.feat_content_layers = {'relu3_2': 1.0}
        else:
            self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
            self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        epsilon = 1e-6
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        temp = featmaps / (reduce_sum + epsilon)
        return temp

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-3
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        div[div<epsilon] = div[div<epsilon] + epsilon
        relative_dist = cdist / div
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        epsilon = 1e-5
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist)/(self.nn_stretch_sigma+ epsilon))
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        epsilon = 1e-5
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / (gen_feats_norm + epsilon)
        tar_normalized = tar_feats / (tar_feats_norm + epsilon)

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)

        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content

        return self.style_loss / gen.size(0) + self.content_loss / gen.size(0)

def gradient_penalty(netD, real_data, fake_data, mask):
    # Random weight for interpolation
    alpha = torch.rand(real_data.size(0), 1, 1, 1, device=real_data.device)
    alpha = alpha.expand_as(real_data)

    # Interpolation
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = Variable(interpolates, requires_grad=True)

    # Discriminator output
    disc_interpolates = netD(interpolates, mask)

    # Gradient computation
    grad_outputs = torch.ones_like(disc_interpolates, device=real_data.device)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

def train(G_net, D_net, device, criterion_pxl, criterion_CE, optimizer_G, optimizer_D, data_loaders, model_save_path, html_save_path, seg_model, n_epochs=500, start_epoch=0):

    id_mrf_loss = IDMRFLoss(device=0)
    measurement = Measurement(3)
    writer = SummaryWriter(os.path.join(html_save_path, 'Seg_result%s' % datetime.now().strftime("%Y%m%d-%H%M%S")))
    seg_model.eval()
    for param in seg_model.parameters():
        param.requires_grad = False

    for epoch in range(start_epoch, n_epochs):

        for phase in ['train', 'val']:
            batches_done = 0
            running_loss_pxl = 0.0
            running_loss_CE = 0.0
            running_loss_MRF = 0.0
            running_metric_MIOU = 0.0
            if phase == 'train':
                running_loss_adv = 0.0
                running_loss_D = 0.0
                running_loss_GP = 0.0

            for idx, (imgs, masked_imgs, seg_mask, b_mask, ob, name) in enumerate(data_loaders[phase]):
                if phase == 'train':
                    G_net.train()
                    D_net.train()

                else:
                    G_net.eval()
                    D_net.eval()
                torch.set_grad_enabled(phase == 'train')

                # Configure input
                imgs = imgs.to(device)
                masked_imgs = masked_imgs.to(device)
                b_mask = b_mask.to(device)
                label_img = mask_labeling(seg_mask, 3).to(device)
                ob = ob.to(device)
                ob = ob*(1-b_mask)

                concat_input = torch.cat(((masked_imgs*2-1), (1-b_mask), ob), dim=1)
                #-----outpainting-----
                outputs = G_net(concat_input)
                #-----segmentation for CE Loss and MIoU-----
                outputs = imgs * b_mask + ((outputs+1)/2) * (1-b_mask)  # [0,1]
                seg_outputs = seg_model(outputs)
                loss_CE = criterion_CE(seg_outputs, label_img)
                seg_blend_np, mask_cpu_np = seg_outputs.detach().cpu().numpy(), label_img.cpu().numpy()
                _, _, _, _, _, _, batch_miou = measurement(seg_blend_np, mask_cpu_np)
                #-----Loss-----
                loss_pxl = criterion_pxl(outputs*2-1, imgs*2-1)
                loss_mrf = id_mrf_loss(outputs, imgs)
                if phase == 'train':
                    fake_scalar = D_net((outputs*2-1).detach(), (1-b_mask))
                    true_scalar = D_net((imgs*2-1), (1-b_mask))
                    w_loss = -torch.mean(true_scalar) + torch.mean(fake_scalar)
                    gp_loss = gradient_penalty(D_net, (imgs*2-1), (outputs*2-1), (1-b_mask))
                    fake_scalar = D_net((outputs*2-1), (1-b_mask))
                    loss_adv = -torch.mean(fake_scalar)
                    #-----weight-----
                    cur_pxl_weight = 1
                    cur_adv_weight = 0.1
                    cur_ce_weight = 0.01
                    cur_mrf_weight = 0.01

                    loss_G = loss_pxl * cur_pxl_weight + cur_adv_weight * loss_adv + cur_ce_weight * loss_CE + cur_mrf_weight * loss_mrf
                    loss_D = w_loss + gp_loss *2

                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()

                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()

                torch.cuda.empty_cache()

                # Update & print statistics
                batches_done += 1
                running_loss_pxl += loss_pxl.item()
                running_loss_CE += loss_CE.item()
                running_loss_MRF += loss_mrf.item()
                running_metric_MIOU += batch_miou.item()
                if phase == 'train':
                    running_loss_adv += loss_adv.item()
                    running_loss_D += loss_D.item()
                    running_loss_GP += gp_loss.item()

                    print('Batch {:d}/{:d}  loss_pxl {:.4f}  loss_adv {:.4f}  loss_D {:.4f}  loss_CE {:.4f} loss_MRF {:.4f} miou {:.4f} GP {:.4f} '.format(
                          batches_done, len(data_loaders[phase]), loss_pxl.item(), loss_adv.item(), loss_D.item(), loss_CE.item(), loss_mrf.item(), batch_miou.item(), gp_loss.item()))

            if phase == 'train':
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(G_net.state_dict(), model_save_path + '/G_' + str(epoch) + '.pt')
                torch.save(D_net.state_dict(), model_save_path + '/D_' + str(epoch) + '.pt')
                generate_html(G_net, D_net, device, data_loaders, html_save_path + '/' + str(epoch))

            # Store & print statistics
            cur_loss_pxl = running_loss_pxl / batches_done
            cur_loss_CE = running_loss_CE / batches_done
            cur_loss_MRF = running_loss_MRF / batches_done
            cur_metric_MIOU = running_metric_MIOU / batches_done
            if phase == 'train':
                cur_loss_adv = running_loss_adv / batches_done
                cur_loss_D = running_loss_D / batches_done
                writer.add_scalars('train/generator_loss',{'Pixel Reconstruction Loss': cur_loss_pxl}, epoch)
                writer.add_scalars('train/generator_loss', {'Adversarial Loss': cur_loss_adv}, epoch)
                writer.add_scalars('train/generator_loss', {'Cross Entropy Loss': cur_loss_CE}, epoch)
                writer.add_scalars('train/generator_loss', {'MRF Loss': cur_loss_MRF}, epoch)
                writer.add_scalars('train/discriminator_loss', {'Discriminator Loss': cur_loss_D}, epoch)
                writer.add_scalars('train/metric',{'MIoU': cur_metric_MIOU}, epoch)
                print('Epoch {:d}/{:d}  {:s}  loss_pxl {:.4f}  loss_adv {:.4f}  loss_D {:.4f}  loss_CE {:.4f} loss_MRF {:.4f} miou {:.4f} '.format(
                        epoch + 1, n_epochs, phase, cur_loss_pxl, cur_loss_adv, cur_loss_D, cur_loss_CE, cur_loss_MRF, cur_metric_MIOU))
            if phase == 'val':
                writer.add_scalars('val/generator_loss',{'Pixel Reconstruction Loss': cur_loss_pxl}, epoch)
                writer.add_scalars('val/generator_loss', {'Cross Entropy Loss': cur_loss_CE}, epoch)
                writer.add_scalars('val/generator_loss', {'MRF Loss': cur_loss_MRF}, epoch)
                writer.add_scalars('val/metric', {'MIoU': cur_metric_MIOU}, epoch)
                print('Epoch {:d}/{:d}  {:s}  loss_pxl {:.4f} loss_CE {:.4f} loss_MRF {:.4f} miou {:.4f} '.format(
                        epoch + 1, n_epochs, phase, cur_loss_pxl, cur_loss_CE, cur_loss_MRF, cur_metric_MIOU))
        print()
    writer.close()
    print('Done!')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def mask_labeling(y_batch: torch.Tensor, num_classes: int) -> torch.Tensor:
    label_pixels = list(torch.unique(y_batch, sorted=True))

    if len(label_pixels) != num_classes:
        print('label pixels error')
        label_pixels = [0, 128, 255]

    for i, px in enumerate(label_pixels):
        y_batch = torch.where(y_batch == px, i, y_batch)

    return y_batch
def pred_to_colormap(pred:np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])): #흰색 파랑 빨강 / 배경 잡초 작물
    pred_label = np.argmax(pred, axis=1) # (N, H, W)
    show_pred = colormap[pred_label] # (N, H, W, 3)
    return show_pred
def save_result_img(input:np.ndarray, target:np.ndarray, pred:np.ndarray, filename, save_dir):
    N = input.shape[0]
    show_pred = pred_to_colormap(pred)
    for i in range(N):
        pred_img = show_pred[i] #(H, W, 3)
        plt.imsave(os.path.join(save_dir, filename[i]), pred_img)