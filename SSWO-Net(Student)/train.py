from typing import List
import os
import yaml
import argparse
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import torch.nn as nn

import torch
import random
from torch.utils.data import DataLoader

import models
from dataset import blind_SegDataset, SegDataset
from metric import Measurement
from loss import DiceLoss

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

class Trainer():
    def __init__(self, opt, cfg, model):
        print(opt)
        print(cfg)
        self.model = model
        self.start_epoch = 0
        self.num_epochs = cfg['NUM_EPOCHS']
        self.device = cfg['GPU']
        self.num_classes = cfg['NUM_CLASSES']

        # data load
        train_dir = os.path.join(cfg['DATA_DIR'], 'train')
        train_dataset = blind_SegDataset(train_dir, resize=cfg['RESIZE'], targetresize=True, direction='top', cover_percent=0.1, randomaug=True) #SegDataset(train_dir, resize=cfg['RESIZE'], targetresize=True, randomaug=True)#
        val_dataset = blind_SegDataset(os.path.join(cfg['DATA_DIR'], 'val'), resize=cfg['RESIZE'], targetresize=True, direction='top', cover_percent=0.1) #SegDataset(os.path.join(cfg['DATA_DIR'], 'val'), resize=cfg['RESIZE'], targetresize=True)#
        self.trainloader = DataLoader(train_dataset, cfg['BATCH_SIZE'], shuffle=True, drop_last=True)
        self.valloader = DataLoader(val_dataset, 1, shuffle=False) #val 데이터는 섞지 않고, 배치사이즈도 1로

        # optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg['OPTIM']['LR_INIT']), betas=(0.5, 0.999))
        # loss function
        #self.loss = DiceLoss(num_classes=cfg['NUM_CLASSES'])
        self.loss = nn.CrossEntropyLoss()
        # # lr scheduler
        # warmup_epochs=3
        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.num_epochs-warmup_epochs, eta_min=1e-7, verbose=True)
        # self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=cosine_scheduler)
        # self.lr_scheduler.step()
        
        self.measurement = Measurement(self.num_classes)
        # if resume
        if cfg['LOAD_WEIGHTS'] != '':
            print('############# resume training #############')
            self.resume = True
            self.device_setting(self.device)
            self.model = self.model.to(self.device)
            #self.mat_model = self.mat_model.load_state_dict(torch.load(opt.mat_weights)['network'])
            #self.mat_model = self.mat_model.to(self.device)
            self.start_epoch, optimizer_statedict, self.best_miou = self.load_checkpoint(cfg['LOAD_WEIGHTS']) #scheduler_statedict,
            self.optimizer.load_state_dict(optimizer_statedict)
            # self.lr_scheduler.load_state_dict(scheduler_statedict)
            
            self.save_dir = os.path.split(os.path.split(cfg['LOAD_WEIGHTS'])[0])[0]
            self.ckpoint_path = os.path.join(self.save_dir, 'ckpoints')
        else:    
            # save path
            self.resume = False
            os.makedirs(cfg['SAVE_DIR'], exist_ok=True)
            train_name = os.path.basename(train_dir)
            fold_name = os.path.basename(cfg['DATA_DIR'])
            self.save_dir = os.path.join(cfg['SAVE_DIR'], f'{self.model.__class__.__name__}-ep{self.num_epochs}-{train_name}-{fold_name}-'+str(len(os.listdir(cfg['SAVE_DIR']))))
            self.ckpoint_path = os.path.join(self.save_dir, 'ckpoints')
            #self.mat_model = self.mat_model.load_state_dict(torch.load(opt.mat_weights)['network'])
            os.makedirs(self.ckpoint_path)
            

    def train(self, opt):
        ###debug###
        # torch.autograd.set_detect_anomaly(True)
        ############
        if not self.resume: 
            self.device_setting(self.device)
            self.model = self.model.to(self.device)
            #self.mat_model.load_state_dict(torch.load(opt.mat_weights)['network'])
            #self.mat_model = self.mat_model.to(self.device)

            self.best_miou = 0
        if opt.save_img:
            os.makedirs(os.path.join(self.save_dir, 'imgs'), exist_ok=True)
        if opt.save_txt:
            self.f = open(os.path.join(self.save_dir, 'result.txt'), 'a')
        if opt.save_graph:
            loss_list = []
            self.val_loss_list = []

        if opt.save_csv:
            loss_list = []
            self.val_loss_list = []
            miou_list, lr_list = [], []
            self.val_miou_list = []
        
        self.best_miou_epoch = 0
        print('######### start training #########')
        for epoch in range(self.start_epoch, self.num_epochs) :
            ep_start = time.time()
            epoch_loss = 0
            epoch_miou = 0
            iou_per_class = np.array([0]*(self.num_classes), dtype=np.float64)
            
            self.model.train()
            #self.mat_model.eval()
            trainloader_len = len(self.trainloader)
            self.start_timer()
            for i, data in enumerate(tqdm(self.trainloader), 0):
                input_img, target_img = data[:2]
                b_mask = data[-2]
                #target_img = torch.where(target_img>=128, 255, 0) #####
                label_img = self.mask_labeling(target_img, self.num_classes)
                input_img, label_img  = input_img.to(self.device), label_img.to(self.device)
                b_mask = b_mask.to(self.device)
                # gradient initialization
                self.optimizer.zero_grad()
                # predict
                concat_input = torch.cat((input_img,b_mask),dim=1)
                #ob = self.mat_model(concat_input)
                #ob = self.pred_to_binary_mask(ob)
                #concat_ob = torch.cat((input_img, b_mask, ob), dim=1)
                pred = self.model(concat_input)
                #pred = self.model(input_img)
                # loss
                loss_output = self.loss(pred, label_img)
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
                
            traintxt = f"[epoch {epoch} Loss: {epoch_loss:.4f}, LearningRate :{self.optimizer.param_groups[0]['lr']:.6f}, trainmIOU: {epoch_miou}, train IOU per class:{epoch_ious}, time: {(time.time()-ep_start):.4f} sec \n" 
                
            print(traintxt)
            if opt.save_txt:
                self.f.write(traintxt)
            # save model
            self.save_checkpoint(epoch, 'model_last.pth')
        
            # validation
            self.val_test(epoch, opt)
            # lr scheduler update
           # self.lr_scheduler.step()
        
        if opt.save_graph:
            self.save_lossgraph(loss_list, self.val_loss_list)
        if opt.save_csv:
            self.save_csv('train', [loss_list, lr_list, miou_list], 'training.csv')
            self.save_csv('val', [self.val_loss_list, self.val_miou_list], 'validation.csv')
            
        print("----- train finish -----")
        self.end_timer_and_print()
            
                
    def device_setting(self, device):
        if device != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda:'+device)
        else: 
            self.device = torch.device('cpu')  
    
    def val_test(self, epoch, opt):
        self.model.eval()
        #self.mat_model.eval()
        val_miou, val_loss = 0, 0
        iou_per_class = np.array([0]*(self.num_classes), dtype=np.float64)
        val_loss_list = []

        for i, data in enumerate(tqdm(self.valloader), 0):
            filename = data[-1]
            input_img, target_img = data[:2]
            #target_img = torch.where(target_img >= 128, 255, 0)  #####
            b_mask = data[-2]
            label_img = self.mask_labeling(target_img, self.num_classes)
            input_img, label_img = input_img.to(self.device), label_img.to(self.device)
            b_mask = b_mask.to(self.device)
            with torch.no_grad():
                concat_input = torch.cat((input_img, b_mask), dim=1)
                #ob = self.mat_model(concat_input)
                #ob = self.pred_to_binary_mask(ob)
                #concat_ob = torch.cat((input_img, b_mask, ob), dim=1)
                pred = self.model(concat_input)
                #pred = self.model(concat_input)
                #pred = self.model(input_img)

                loss_output = self.loss(pred, label_img)
                val_loss += loss_output.item()

            pred_numpy, label_numpy = pred.detach().cpu().numpy(), label_img.detach().cpu().numpy()
            _, ep_miou, ious, _, _, _ = self.measurement(pred_numpy, label_numpy)
            val_miou += ep_miou
            iou_per_class += ious
            if opt.save_img:
                self.save_result_img(input_img.detach().cpu().numpy(), \
                    target_img.detach().cpu().numpy(), pred_numpy, filename, os.path.join(self.save_dir, 'imgs'))


        val_miou = val_miou / len(self.valloader)
        val_ious = np.round((iou_per_class / len(self.valloader)), 5).tolist()
        val_loss = val_loss / len(self.valloader)
        val_loss_list.append(val_loss)

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
            self.val_loss_list += [val_loss]

    def save_checkpoint(self, epoch, filename):
        filename = os.path.join(self.ckpoint_path, filename)
        torch.save({'network':self.model.state_dict(),
                    'epoch': epoch,
                    'optimizer':self.optimizer.state_dict(),
                    # 'scheduler':self.lr_scheduler.state_dict(),
                    'best_miou':self.best_miou,},
                    filename)

    def load_checkpoint(self, weights_path, istrain=True):
        chkpoint = torch.load(weights_path)
        self.model.load_state_dict(chkpoint['network'])
        if istrain:
            return chkpoint['epoch'], chkpoint['optimizer'], chkpoint['best_miou'] # chkpoint['scheduler'],
        
    def mask_labeling(self, y_batch:torch.Tensor, num_classes:int):
        label_pixels = list(torch.unique(y_batch, sorted=True))
        assert len(label_pixels) <= num_classes, 'too many label pixels'
        if len(label_pixels) < num_classes:
            print('label pixels error')
            label_pixels = [0, 128, 255]
        
        for i, px in enumerate(label_pixels):
            y_batch = torch.where(y_batch==px, i, y_batch)

        return y_batch

    def pred_to_colormap(self, pred:np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])):
        pred_label = np.argmax(pred, axis=1) # (N, H, W)
        show_pred = colormap[pred_label] # (N, H, W, 3)
        return show_pred

    def pred_to_binary_mask(self, pred: torch.Tensor) -> torch.Tensor:
        pred_binary = torch.argmax(pred, dim=1, keepdim=True)

        return pred_binary.float()

    def save_result_img(self, input:np.ndarray, target:np.ndarray, pred:np.ndarray, filename, save_dir):
        N = input.shape[0]
        show_pred = self.pred_to_colormap(pred)
        for i in range(N):
            input_img = np.transpose(input[i], (1, 2, 0)) # (H, W, 3)
            target_img = np.transpose(np.array([target[i]/255]*3), (1, 2, 0)) # (3, H, W) -> (H, W, 3)
            pred_img = show_pred[i] #(H, W, 3)
            cat_img = np.concatenate((input_img, target_img, pred_img), axis=1) # (H, 3W, 3)
            plt.imsave(os.path.join(save_dir, filename[i]), cat_img)
    
    def save_lossgraph(self, train_loss:list, val_loss:list):
        # the graph for Loss
        plt.figure(figsize=(10,5))
        plt.title("Loss")
        plt.plot(train_loss, label='Train loss')
        plt.plot(val_loss, label='Validation loss')
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.legend() # 범례
        plt.savefig(os.path.join(self.save_dir, 'Loss_Graph.png'))
    
    def save_csv(self, mode, value_list:List, filename):
        if mode=='train':
            df = pd.DataFrame({'loss':value_list[0],
                                'lr':value_list[1],
                                'miou':value_list[2]
                                })
        if mode=='val':
            df = pd.DataFrame({'val_loss':value_list[0],
                               'val_miou':value_list[1]})
            
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
    parser.add_argument('--model', type=str, default='munet', help='segmentation model''s name for training')
    parser.add_argument('--config', type=str, default='./config/semantic_segmentation_without_outpainting_config.yaml', help='yaml file that has segmentation train config information')
    parser.add_argument('--save_img', type=bool, default=True, help='save result images')
    parser.add_argument('--save_txt', type=bool, default=True, help='save training process as txt file')
    parser.add_argument('--save_csv', type=bool, default=True, help='save training process as csv file')
    parser.add_argument('--save_graph', type=bool, default=True, help='save Loss graph with plt')
    opt = parser.parse_args()

    
    if opt.model == 'unet':
        model = models.Unet(in_channels=5)
        mat_model = models.Unet(in_channels=4, num_classes=2)

    if opt.model == 'dwsunet':
        model = models.DWSUnet(in_channels=5)
        mat_model = models.DWSUnet(in_channels=4, num_classes=2)

    if opt.model == 'deeplabv3plus':
        model = models.Resnet50_DeepLabv3Plus()
        
    if opt.model == 'segnet':
        model = models.SegNet(3, 512, 3)

    if opt.model == 'cgnet':
        model = models.CGNet(3, 3)

    if opt.model == 'myunet':
        model = models.myUNET()

    if opt.model == 'fcn':
        #vgg_model = models.VGGNet(requires_grad=True)
        model = models.FCN(num_classes=3)

    if opt.model == 'munet':
        #vgg_model = models.VGGNet(requires_grad=True)
        model = models.MobileUNet(in_channels=4)
        
    with open(opt.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    trainer = Trainer(opt, cfg, model)
    trainer.train(opt)
