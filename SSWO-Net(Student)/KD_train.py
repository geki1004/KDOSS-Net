from typing import List
import os

import functorch.dim
import yaml
import argparse
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import torch.nn as nn
from collections import defaultdict
import torch
import random
from torch.utils.data import DataLoader, Subset

import models
from dataset import semantic_segmentation_Dataset, blind_semantic_segmentation_Dataset, KD_semantic_segmentation_Dataset
from metric import Measurement
import torch.nn.functional as F

# 시드 설정
seed_value = 123

# PyTorch 시드 설정
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # GPU 사용 시 CUDA 시드 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Python 시드 설정
random.seed(seed_value)

# Numpy 시드 설정
np.random.seed(seed_value)

class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Args:
            in_channels (int): Input channel size.
            reduction_ratio (int): Reduction ratio for channel squeeze.
        """
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        #self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)  # Squeeze
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        #self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)  # Excitation
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Scale

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Global Average Pooling
        se = self.global_avg_pool(x)#.view(batch_size, channels)
        # Fully connected layers
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)#.view(batch_size, channels, 1, 1)  # Reshape to match input dimensions
        # Scale the input feature map
        return x * se.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling

        # MLP를 통해 채널 중요도 계산
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)  # 차원 축소
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)  # 차원 복원
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # GAP와 GMP을 각각 계산
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        # Flatten을 통해 (batch_size, channels, 1, 1) -> (batch_size, channels)로 변환
        avg_out = self.flatten(avg_out)
        max_out = self.flatten(max_out)

        # 두 pooling 결과를 MLP로 처리하여 채널 중요도 계산
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        # 채널 중요도를 결합하고 Sigmoid로 정규화
        attention = self.sigmoid(avg_out + max_out)

        # attention을 입력에 곱하여 중요한 채널을 강조
        return x * attention.view(attention.size(0), attention.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))

class CBAM(nn.Module):
    def __init__(self, in_channels, kernel_size=7, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        """
        Self-Attention Module
        Args:
            in_dim (int): Input dimension (e.g., number of channels)
        """
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)  # Apply softmax on the last dimension

    def forward(self, x):
        """
        Forward pass of the Self-Attention layer.
        Args:
            x (Tensor): Input feature map of shape (B, C, H, W)

        Returns:
            attention_map (Tensor): Attention map of shape (B, H*W, H*W)
            out (Tensor): Output feature map of shape (B, C, H, W)
        """
        B, C, H, W = x.size()

        # Compute query, key, and value
        query = self.query(x).view(B, -1, H * W)  # Shape: (B, C', H*W)
        key = self.key(x).view(B, -1, H * W)  # Shape: (B, C', H*W)
        value = self.value(x).view(B, -1, H * W)  # Shape: (B, C, H*W)

        # Compute attention map
        attention_map = torch.bmm(query.permute(0, 2, 1), key)  # Shape: (B, H*W, H*W)
        attention_map = self.softmax(attention_map)  # Normalize

        # Apply attention map to value
        out = torch.bmm(value, attention_map.permute(0, 2, 1))  # Shape: (B, C, H*W)
        out = out.view(B, C, H, W)  # Reshape to original spatial dimensions

        return out

class contextual_consistency_loss(nn.Module):
    def __init__(self):
        super(contextual_consistency_loss, self).__init__()

    def forward(self, feature_S, feature_T, mask):
        loss = self.contextual_consistency_loss(feature_S, feature_T, mask)
        return loss

    def contextual_consistency_loss(self, student_features, teacher_features, mask, reduction='mean'):
        """
        student_features: Student의 Feature Map (B, C, H, W)
        teacher_features: Teacher의 Feature Map (B, C, H, W)
        mask: 가려진 영역의 Binary Mask (B, 1, H, W)
        reduction: 'mean' 또는 'sum' (손실 계산 방식)
        """
        # 복원된 영역 선택
        student_inpainted = student_features * mask
        teacher_inpainted = teacher_features * mask

        # 주변 영역 선택
        student_context = student_features * (1 - mask)
        teacher_context = teacher_features * (1 - mask)

        # 채널 평균을 통해 공간적 유사도 계산
        student_similarity = F.cosine_similarity(student_inpainted, student_context, dim=1)
        teacher_similarity = F.cosine_similarity(teacher_inpainted, teacher_context, dim=1)

        # Teacher와 Student 간 유사도 차이를 최소화
        loss = F.mse_loss(student_similarity, teacher_similarity, reduction=reduction)
        return loss
class FSP(nn.Module):
	'''
	A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
	http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
	'''
	def __init__(self):
		super(FSP, self).__init__()

	def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
		loss = F.mse_loss(self.fsp_matrix(fm_s1,fm_s2), self.fsp_matrix(fm_t1,fm_t2))

		return loss

	def fsp_matrix(self, fm1, fm2):
		if fm1.size(2) > fm2.size(2):
			fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

		fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
		fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

		fsp = torch.bmm(fm1, fm2) / fm1.size(2)

		return fsp
class MLP(nn.Sequential):
    '''
    double 3x3 conv layers with Batch normalization and ReLU
    '''

    def __init__(self, in_channels, out_channels, kerner_size=1):
        conv_layers = [
            nn.Conv2d(in_channels, out_channels, kerner_size),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kerner_size),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
        ]
        super(MLP, self).__init__(*conv_layers)
class Trainer():
    def __init__(self, opt, cfg, model_T, model_S):
        print(opt)
        print(cfg)
        self.model_T = model_T
        self.model_S = model_S
        self.start_epoch = 0
        self.num_epochs = cfg['NUM_EPOCHS']
        self.device = cfg['GPU']
        self.num_classes = cfg['NUM_CLASSES']
        self.KD_weight = 0.3
        #self.self = SelfAttention(in_dim=1024)
        #self.StudentTransBlock = SEBlock(in_channels=1024, reduction_ratio=16)
        #self.StudentTransBlock = MLP(in_channels=1024, out_channels=1024)
        #self.StudentTransBlock2 = MLP(in_channels=64, out_channels=64)
        #self.StudentTransBlock = ChannelAttention(in_channels=1024, reduction_ratio=16)
        #self.StudentTransBlock = CBAM(in_channels=1024, reduction_ratio=16)
        self.StudentTransBlock2 = CBAM(in_channels=64, reduction_ratio=16)

        # data load
        train_dir = os.path.join(cfg['DATA_DIR'], 'train')
        train_dataset = KD_semantic_segmentation_Dataset(train_dir, resize=cfg['RESIZE'], targetresize=True, direction='top', cover_percent=0.1, randomaug=None)  # SegDataset(train_dir, resize=cfg['RESIZE'], targetresize=True, randomaug=True)#
        val_dataset = KD_semantic_segmentation_Dataset(os.path.join(cfg['DATA_DIR'], 'test'), resize=cfg['RESIZE'], targetresize=True, direction='top', cover_percent=0.1)  # SegDataset(os.path.join(cfg['DATA_DIR'], 'val'), resize=cfg['RESIZE'], targetresize=True)#
        self.trainloader = DataLoader(train_dataset, cfg['BATCH_SIZE'], shuffle=True, drop_last=True)
        self.valloader = DataLoader(val_dataset, 1, shuffle=False)  # val 데이터는 섞지 않고, 배치사이즈도 1로

        # optimizer
        self.optimizer = torch.optim.Adam([{'params' : model_S.parameters()}], lr=float(cfg['OPTIM']['LR_INIT']), betas=(0.5, 0.999))
        #self.optimizer.add_param_group({'params': self.StudentTransBlock.parameters()})
        self.optimizer.add_param_group({'params': self.StudentTransBlock2.parameters()})
        # loss function
        self.loss_task = nn.CrossEntropyLoss()
        self.loss_KD = nn.MSELoss()
        self.loss_KD_last = nn.MSELoss()
        self.loss_KD_output = nn.KLDivLoss()
        self.temperature = 3

        self.measurement = Measurement(self.num_classes)
        # if resume
        if cfg['LOAD_WEIGHTS'] != '':
            print('############# resume training #############')
            self.resume = True
            self.device_setting(self.device)
            self.model_S = self.model_S.to(self.device)
            self.model_T = self.model_T.load_state_dict(torch.load(opt.weights_T)['network'])
            self.model_T = self.model_T.to(self.device)
            #self.StudentTransBlock = self.StudentTransBlock.to(self.device)
            self.StudentTransBlock2 = self.StudentTransBlock2.to(self.device)
            self.start_epoch, optimizer_statedict, self.best_miou = self.load_checkpoint(cfg['LOAD_WEIGHTS'])  # scheduler_statedict,
            self.optimizer.load_state_dict(optimizer_statedict)

            self.save_dir = os.path.split(os.path.split(cfg['LOAD_WEIGHTS'])[0])[0]
            self.ckpoint_path = os.path.join(self.save_dir, 'ckpoints')
        else:
            # save path
            self.resume = False
            os.makedirs(cfg['SAVE_DIR'], exist_ok=True)
            train_name = os.path.basename(train_dir)
            fold_name = os.path.basename(cfg['DATA_DIR'])
            self.save_dir = os.path.join(cfg['SAVE_DIR'],
                                         f'{self.model_S.__class__.__name__}-ep{self.num_epochs}-{train_name}-{fold_name}-' + str(
                                             len(os.listdir(cfg['SAVE_DIR']))))
            self.ckpoint_path = os.path.join(self.save_dir, 'ckpoints')
            os.makedirs(self.ckpoint_path)

    def train(self, opt):
        ###debug###
        # torch.autograd.set_detect_anomaly(True)
        ############
        if not self.resume:
            self.device_setting(self.device)
            self.model_S = self.model_S.to(self.device)
            self.model_T.load_state_dict(torch.load(opt.weights_T)['network'])
            self.model_T = self.model_T.to(self.device)
            #self.StudentTransBlock = self.StudentTransBlock.to(self.device)
            self.StudentTransBlock2 = self.StudentTransBlock2.to(self.device)

            self.best_miou = 0
        if opt.save_img:
            os.makedirs(os.path.join(self.save_dir, 'imgs'), exist_ok=True)
        if opt.save_txt:
            self.f = open(os.path.join(self.save_dir, 'result.txt'), 'a')
        if opt.save_graph:
            loss_list = []
            self.val_loss_task_list = []
            self.val_loss_KD_list = []

        if opt.save_csv:
            loss_list = []
            self.val_loss_task_list = []
            self.val_loss_KD_list = []
            miou_list, lr_list = [], []
            self.val_miou_list = []

        self.best_miou_epoch = 0
        print('######### start training #########')
        for epoch in range(self.start_epoch, self.num_epochs):
            ep_start = time.time()
            epoch_loss = 0
            epoch_miou = 0
            iou_per_class = np.array([0] * (self.num_classes), dtype=np.float64)

            self.model_S.train()
            self.model_T.eval()
            #self.StudentTransBlock.train()
            self.StudentTransBlock2.train()
            trainloader_len = len(self.trainloader)
            self.start_timer()
            for i, data in enumerate(tqdm(self.trainloader), 0):
                input_img, target_img = data[:2]
                b_mask = data[-3]
                input_T = data[-2]
                label_img = self.mask_labeling(target_img, self.num_classes)
                input_img, label_img = input_img.to(self.device), label_img.to(self.device)
                b_mask = b_mask.to(self.device)
                input_T = input_T.to(self.device)

                # predict
                with torch.no_grad():
                    pred_T , _ , feature_T_last = self.model_T(input_T)

                concat_input = torch.cat((input_img, b_mask), dim=1)
                pred, _, feature_S_last = self.model_S(concat_input)
                #feature_S_mid_b = self.StudentTransBlock(feature_S_mid)
                feature_S_last_b = self.StudentTransBlock2(feature_S_last)

                softmax_S = F.log_softmax(pred/self.temperature, dim=1)
                softmax_T = F.softmax(pred_T/self.temperature, dim=1)
                # loss

                self.optimizer.zero_grad()
                loss_output = (1 - self.KD_weight) * self.loss_task(pred, label_img) + self.KD_weight * (self.loss_KD_output(softmax_S, softmax_T) + self.loss_KD_last(feature_S_last_b, feature_T_last)) #+ self.loss_KD(feature_S_mid, feature_T_mid) + self.loss_KD_last(feature_S_last, feature_T_last)
                loss_output.backward()
                # update
                self.optimizer.step()

                pred_numpy, label_numpy = pred.detach().cpu().numpy(), label_img.detach().cpu().numpy()
                epoch_loss += loss_output.item()
                _, ep_miou, ious, _, _, _ = self.measurement(pred_numpy, label_numpy)
                epoch_miou += ep_miou
                iou_per_class += ious
            epoch_loss /= trainloader_len
            epoch_miou /= trainloader_len
            epoch_ious = np.round((iou_per_class / trainloader_len), 5).tolist()

            if opt.save_graph:
                loss_list += [epoch_loss]
            if opt.save_csv:
                if not opt.save_graph: loss_list += [epoch_loss]
                miou_list += [epoch_miou]
                lr_list += [self.optimizer.param_groups[0]['lr']]

            traintxt = f"[epoch {epoch} Loss: {epoch_loss:.4f}, LearningRate :{self.optimizer.param_groups[0]['lr']:.6f}, trainmIOU: {epoch_miou}, train IOU per class:{epoch_ious}, time: {(time.time() - ep_start):.4f} sec \n"

            print(traintxt)
            if opt.save_txt:
                self.f.write(traintxt)
            # save model
            self.save_checkpoint(epoch, 'model_last.pth')
            torch.cuda.empty_cache()
            # validation
            self.val_test(epoch, opt)

        if opt.save_graph:
            self.save_lossgraph(loss_list, self.val_loss_task_list, self.val_loss_KD_list)
        if opt.save_csv:
            self.save_csv('train', [loss_list, lr_list, miou_list], 'training.csv')
            self.save_csv('val', [self.val_loss_task_list, self.val_loss_KD_list, self.val_miou_list], 'validation.csv')

        print("----- train finish -----")
        self.end_timer_and_print()

    def device_setting(self, device):
        if device != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda:' + device)
        else:
            self.device = torch.device('cpu')

    def val_test(self, epoch, opt):
        self.model_T.eval()
        self.model_S.eval()
        #self.StudentTransBlock.eval()
        self.StudentTransBlock2.eval()
        val_miou, val_loss_task, val_loss_KD = 0, 0, 0
        iou_per_class = np.array([0] * (self.num_classes), dtype=np.float64)
        val_loss_task_list, val_loss_KD_list = [], []

        for i, data in enumerate(tqdm(self.valloader), 0):
            filename = data[-1]
            input_img, target_img = data[:2]
            # target_img = torch.where(target_img >= 128, 255, 0)  #####
            b_mask = data[-3]
            input_T = data[-2]
            label_img = self.mask_labeling(target_img, self.num_classes)
            input_img, label_img = input_img.to(self.device), label_img.to(self.device)
            b_mask = b_mask.to(self.device)
            input_T = input_T.to(self.device)
            with torch.no_grad():
                concat_input = torch.cat((input_img, b_mask), dim=1)
                pred, _,feature_S_last = self.model_S(concat_input)
                pred_T, _, feature_T_last = self.model_T(input_T)
                #pred, _, _ = self.model_S(concat_input)
                #pred_T, _, _ = self.model_T(input_T)
                #feature_S_mid_b = self.StudentTransBlock(feature_S_mid)
                feature_S_last_b = self.StudentTransBlock2(feature_S_last)

                softmax_S = F.log_softmax(pred / self.temperature, dim=1)
                softmax_T = F.softmax(pred_T / self.temperature, dim=1)
                loss_task = self.loss_task(pred, label_img)
                loss_KD = self.loss_KD_output(softmax_S, softmax_T) + self.loss_KD_last(feature_S_last_b, feature_T_last) #+ self.loss_KD(feature_S_mid,feature_T_mid)+ self.loss_KD(feature_S_mid,feature_T_mid)+ self.loss_KD_last(feature_S_last, feature_T_last)
                val_loss_task += loss_task.item()
                val_loss_KD += loss_KD.item()

            pred_numpy, label_numpy = pred.detach().cpu().numpy(), label_img.detach().cpu().numpy()
            _, ep_miou, ious, _, _, _ = self.measurement(pred_numpy, label_numpy)
            val_miou += ep_miou
            iou_per_class += ious
            if opt.save_img:
                self.save_result_img(input_img.detach().cpu().numpy(), \
                                     target_img.detach().cpu().numpy(), pred_numpy, filename,
                                     os.path.join(self.save_dir, 'imgs'))

        val_miou = val_miou / len(self.valloader)
        val_ious = np.round((iou_per_class / len(self.valloader)), 5).tolist()
        val_loss_task = val_loss_task / len(self.valloader)
        val_loss_KD = val_loss_KD / len(self.valloader)
        val_loss_task_list.append(val_loss_task)
        val_loss_KD_list.append(val_loss_KD)

        if val_miou >= self.best_miou:
            self.best_miou = val_miou
            self.best_miou_epoch = epoch
            self.save_checkpoint(epoch, 'best_miou.pth')

        valtxt = f"[val][epoch {epoch} mIOU: {val_miou:.4f}, IOU per class:{val_ious}---best mIOU:{self.best_miou}, best mIOU epoch: {self.best_miou_epoch}]\n"
        print(valtxt)
        # best miou model save
        if opt.save_txt:
            self.f.write(valtxt)
        if opt.save_csv:
            self.val_miou_list += [val_miou]
            self.val_loss_task_list += [val_loss_task]
            self.val_loss_KD_list += [val_loss_KD]

    def save_checkpoint(self, epoch, filename):
        filename = os.path.join(self.ckpoint_path, filename)
        torch.save({'network': self.model_S.state_dict(),
                    'epoch': epoch,
                    'optimizer': self.optimizer.state_dict(),
                    # 'scheduler':self.lr_scheduler.state_dict(),
                    'best_miou': self.best_miou, },
                   filename)

    def load_checkpoint(self, weights_path, istrain=True):
        chkpoint = torch.load(weights_path)
        self.model_S.load_state_dict(chkpoint['network'])
        if istrain:
            return chkpoint['epoch'], chkpoint['optimizer'], chkpoint['best_miou']  # chkpoint['scheduler'],

    def mask_labeling(self, y_batch: torch.Tensor, num_classes: int):
        label_pixels = list(torch.unique(y_batch, sorted=True))
        assert len(label_pixels) <= num_classes, 'too many label pixels'
        if len(label_pixels) < num_classes:
            print('label pixels error')
            label_pixels = [0, 128, 255]

        for i, px in enumerate(label_pixels):
            y_batch = torch.where(y_batch == px, i, y_batch)

        return y_batch

    def pred_to_colormap(self, pred: np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])):
        pred_label = np.argmax(pred, axis=1)  # (N, H, W)
        show_pred = colormap[pred_label]  # (N, H, W, 3)
        return show_pred

    def pred_to_binary_mask(self, pred: torch.Tensor) -> torch.Tensor:
        pred_binary = torch.argmax(pred, dim=1, keepdim=True)

        return pred_binary.float()

    def save_result_img(self, input: np.ndarray, target: np.ndarray, pred: np.ndarray, filename, save_dir):
        N = input.shape[0]
        show_pred = self.pred_to_colormap(pred)
        for i in range(N):
            input_img = np.transpose(input[i], (1, 2, 0))  # (H, W, 3)
            target_img = np.transpose(np.array([target[i] / 255] * 3), (1, 2, 0))  # (3, H, W) -> (H, W, 3)
            pred_img = show_pred[i]  # (H, W, 3)
            cat_img = np.concatenate((input_img, target_img, pred_img), axis=1)  # (H, 3W, 3)
            plt.imsave(os.path.join(save_dir, filename[i]), cat_img)

    def save_lossgraph(self, train_loss: list, val_loss_task: list, val_loss_KD: list):
        # the graph for Loss
        plt.figure(figsize=(10, 5))
        plt.title("Loss")
        plt.plot(train_loss, label='Train loss')
        plt.plot(val_loss_task, label='Validation loss_task')
        plt.plot(val_loss_KD, label='Validation loss_KD')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend()  # 범례
        plt.savefig(os.path.join(self.save_dir, 'Loss_Graph.png'))

    def save_csv(self, mode, value_list: List, filename):
        if mode == 'train':
            df = pd.DataFrame({'loss': value_list[0],
                               'lr': value_list[1],
                               'miou': value_list[2]
                               })
        if mode == 'val':
            df = pd.DataFrame({'val_loss_task': value_list[0],
                               'val_loss_KD': value_list[1],
                               'val_miou': value_list[2]})

        df.to_csv(os.path.join(os.path.abspath(self.save_dir), filename), mode='a')

    def start_timer(self):
        '''before training processes'''
        global start_time
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
        start_time = time.time()

    def end_timer_and_print(self):
        torch.cuda.synchronize()
        end_time = time.time()
        print("Total execution time = {:.3f} sec".format(end_time - start_time))
        print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet_munet', help='segmentation model''s name for training')
    parser.add_argument('--config', type=str, default='./config/object_prediction_config.yaml',help='yaml file that has segmentation train config information')
    parser.add_argument('--save_img', type=bool, default=True, help='save result images')
    parser.add_argument('--save_txt', type=bool, default=True, help='save training process as txt file')
    parser.add_argument('--save_csv', type=bool, default=True, help='save training process as csv file')
    parser.add_argument('--save_graph', type=bool, default=True, help='save Loss graph with plt')
    parser.add_argument('--weights_T', type=str, default='C:/Users/shc01/Downloads/weight/bonirob/seg/Unet-ep200-train-test-gated-1/ckpoints/best_miou.pth', help='weights file for test')
    opt = parser.parse_args()

    if opt.model == 'unet_munet':
        model_T = models.Unet(in_channels=3, num_classes=3, first_outchannels=64)
        model_S = models.MobileUNet(in_channels=4, num_classes=3)

    with open(opt.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    trainer = Trainer(opt, cfg, model_T, model_S)
    trainer.train(opt)
