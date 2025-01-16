import os
import argparse
import gc
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import models
from dataset import blind_SegDataset, SegDataset, mask_SegDataset
from metric import Measurement
def visualize_feature_maps(feature_maps, path_name):
    num_feature_maps = len(feature_maps)
    for i, fmap in enumerate(feature_maps):
        fmap = fmap.detach().cpu().numpy()  # 텐서를 numpy 배열로 변환
        num_channels = fmap.shape[1]
        size = min(8, num_channels)  # 한번에 최대 8개의 채널만 시각화

        fig, axes = plt.subplots(1, size, figsize=(15, 15))
        fig.suptitle(f'{path_name} Layer {i+1}')
        for j in range(size):
            if size == 1:
                ax = axes
            else:
                ax = axes[j]
            ax.imshow(fmap[0, j], cmap='gray')
            ax.axis('off')
        plt.show()
def mask_labeling(y_batch:torch.Tensor, num_classes:int) -> torch.Tensor:
    label_pixels = list(torch.unique(y_batch, sorted=True))
    
    if len(label_pixels) != num_classes:
        print('label pixels error')
        label_pixels = [0, 128, 255]
    
    for i, px in enumerate(label_pixels):
        y_batch = torch.where(y_batch==px, i, y_batch)

    return y_batch

def pred_to_colormap(pred:np.ndarray, colormap=np.array([[0., 0., 0.], [0., 0., 1.], [1., 0., 0.]])): #흰색 파랑 빨강 / 배경 잡초 작물
    pred_label = np.argmax(pred, axis=1) # (N, H, W)
    show_pred = colormap[pred_label] # (N, H, W, 3)
    return show_pred


def save_result_img(input:np.ndarray, target:np.ndarray, pred:np.ndarray, filename, save_dir):
    N = input.shape[0]
    show_pred = pred_to_colormap(pred)
    for i in range(N):
        input_img = np.transpose(input[i], (1, 2, 0)) # (H, W, 3)
        target_img = np.transpose(np.array([target[i]/255]*3), (1, 2, 0)) # (3, H, W) -> (H, W, 3)
        pred_img = show_pred[i] #(H, W, 3)
        #r = np.where(pred_img == 1, 1, 0)
        #r = np.repeat(r[:, :, :1], 3, axis=2)

        #input_img = input_img*r
        cat_img = np.concatenate((input_img, target_img, pred_img), axis=1) # (H, 3W, 3)
        #plt.imsave(os.path.join(save_dir, filename[i]), pred_img)
        plt.imsave(os.path.join(save_dir, filename[i]), cat_img)
        #plt.imsave(os.path.join(save_dir, filename[i]), input_img)
    
def test(model, opt):
    torch.cuda.empty_cache()
    gc.collect()
    print(opt)
    
    test_data = SegDataset(opt.data_dir, resize=512, targetresize=True) #blind_SegDataset(opt.data_dir, resize=512, targetresize=True, direction='top', cover_percent=0.1) #
    testloader = DataLoader(test_data, 1, shuffle=False)
    device = torch.device('cuda:'+opt.gpu) if opt.gpu != '-1' else torch.device('cpu')
    is_rst = opt.data_dir.split('/')[-1]
    is_rst2 = opt.data_dir.split('/')[-2]
    save_dir = os.path.join(opt.save_dir, f'{model.__class__.__name__}-{is_rst}-{is_rst2}-' + str(len(os.listdir(opt.save_dir))))
    os.makedirs(save_dir)
    
    num_classes = opt.num_classes
    measurement = Measurement(num_classes)
    # model = load_checkpoint(model, opt.weights)
    print('load weights...')
    try:
        model.load_state_dict(torch.load(opt.weights)['network'])
    except:
        model.load_state_dict(torch.load(opt.weights))
    model = model.to(device)
    if opt.save_txt:
        f = open(os.path.join(save_dir, 'results.txt'), 'w')
        f.write(f"data_dir:{opt.data_dir}, weights:{opt.weights}, save_dir:{opt.save_dir}")
    if opt.save_img:
        os.mkdir(os.path.join(save_dir, 'imgs'))
    
    model.eval()
    test_acc, test_miou = 0, 0
    test_precision, test_recall, test_f1score = 0, 0, 0
    iou_per_class = np.array([0]*(opt.num_classes), dtype=np.float64)
    for input_img, mask_img,  filename in tqdm(testloader):#b_mask,
        input_img= input_img.to(device)
        #b_mask= b_mask.to(device)
        mask_cpu = mask_labeling(mask_img.type(torch.long), opt.num_classes)
        with torch.no_grad():
            #concat_input = torch.cat((input_img,b_mask),dim=1)
            #pred = model(concat_input) #, feature
            pred = model(input_img)  # , feature
            # ContractingPath 피처 맵 시각화
            #visualize_feature_maps(feature['contracting_features'], 'Contracting Path')

            # ExpansivePath 피처 맵 시각화
            #visualize_feature_maps(feature['expansive_features'], 'Expansive Path')
        pred = F.interpolate(pred, mask_img.shape[-2:], mode='bilinear')
        pred_cpu, mask_cpu = pred.detach().cpu().numpy(), mask_cpu.cpu().numpy()
        acc_pixel, batch_miou, iou_ndarray, precision, recall, f1score = measurement(pred_cpu, mask_cpu)

        test_acc += acc_pixel
        test_miou += batch_miou
        iou_per_class += iou_ndarray
        
        test_precision += precision
        test_recall += recall
        test_f1score += f1score
            
        if opt.save_img:
            input_img = F.interpolate(input_img.detach().cpu(), mask_img.shape[-2:], mode='bilinear')
            #input_img = np.where(pred.detach().cpu()>0, input_img, 0)
            save_result_img(input_img.numpy(), mask_img.detach().cpu().numpy(), pred.detach().cpu().numpy(), filename, os.path.join(save_dir, 'imgs'))
    
    # test finish
    test_acc = test_acc / len(testloader)
    test_miou = test_miou / len(testloader)
    test_ious = np.round((iou_per_class / len(testloader)), 5).tolist()
    test_precision /= len(testloader)
    test_recall /= len(testloader)
    test_f1score /= len(testloader)
    
    result_txt = "load model(.pt) : %s \n Testaccuracy: %.8f, Test miou: %.8f" % (opt.weights,  test_acc, test_miou)       
    result_txt += f"\niou per class {test_ious}"
    result_txt += f"\nprecision : {test_precision}, recall : {test_recall}, f1score : {test_f1score} "
    print(result_txt)
    if opt.save_txt:
        f.write(result_txt)
        f.close()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser() #
    parser.add_argument('--data_dir', type=str, default='C:/Users/8138/Downloads/data/cropweed_total/IJRR2017/occ/gated2/test', help='directory that has data')#C:/Users/shc/Downloads/data/cropweed_total/IJRR2017/occ/1/test
    parser.add_argument('--save_dir', type=str, default='D:/save/rice/seg', help='directory for saving results')
    parser.add_argument('--weights', type=str, default='D:/save/bonirob/seg/Unet-ep200-train-gated2-16/ckpoints/best_miou.pth', help='weights file for test') #C:/Users/shc/Downloads/save/seg/good/Unet-ep400-train_mat-1-ce/ckpoints/model_last.pth C:/Users/shc/Downloads/save/seg/good/Unet-ep400-train-1-512/ckpoints/best_miou.pth
    parser.add_argument('--save_img', type=bool, default=True, help='save result images')
    parser.add_argument('--save_txt', type=bool, default=True, help='save training process as txt file')
    parser.add_argument('--show_img', type=bool, default=False, help='show images')
    parser.add_argument('--gpu', type=str, default='0', help='gpu number. -1 is cpu')
    ##############
    parser.add_argument('--model', type=str, default='unet', help='modelname')
    parser.add_argument('--num_classes', type=int, default=3, help='the number of classes')
    
    opt = parser.parse_args()
    assert opt.model in ['unet', 'deeplabv3', 'segnet', 'cgnet', 'myunet', 'fcn'], 'opt.model is not available'
    
    if opt.model == 'unet':
        model = models.Unet(in_channels=3)
    elif opt.model == 'deeplabv3':
        model = models.Resnet50_DeepLabv3Plus()
    elif opt.model == 'segnet':
        model = models.SegNet(3, 512, 3)
    elif opt.model == 'cgnet':
        model = models.CGNet(3, 3)
    elif opt.model == 'myunet':
        model = models.myUNET()
    elif opt.model == 'fcn':
        model = models.FCN(num_classes=3)
    test(model, opt)

   