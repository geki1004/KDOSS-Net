import os
import argparse
import gc
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import models
from dataset import object_prediction_Dataset
from metric import Measurement

def mask_labeling(y_batch:torch.Tensor, num_classes:int) -> torch.Tensor:
    label_pixels = list(torch.unique(y_batch, sorted=True))
    
    if len(label_pixels) != num_classes:
        print('label pixels error')
        label_pixels = [0, 128, 255]
    
    for i, px in enumerate(label_pixels):
        y_batch = torch.where(y_batch==px, i, y_batch)

    return y_batch

def pred_to_colormap(pred:np.ndarray, colormap=np.array([[0., 0., 0.], [1., 1., 1.], [1., 0., 0.]])): #흰색 파랑 빨강 / 배경 잡초 작물
    pred_label = np.argmax(pred, axis=1) # (N, H, W)
    show_pred = colormap[pred_label] # (N, H, W, 3)
    return show_pred

def pred_to_binary_mask(pred: torch.Tensor) -> torch.Tensor:
    pred_binary = torch.argmax(pred, dim=1, keepdim=True)

    return pred_binary.float()

def save_result_img(input:np.ndarray, target:np.ndarray, pred:np.ndarray, ob:np.ndarray, b_mask:np.ndarray, filename, save_dir):
    N = input.shape[0]
    show_pred = pred_to_colormap(pred)
    ob = ob*255
    for i in range(N):
        input_img = np.transpose(input[i], (1, 2, 0)) # (H, W, 3)

        input_dir = os.path.join(save_dir, 'input')
        target_dir = os.path.join(save_dir, 'target')
        ob_dir = os.path.join(save_dir, 'ob')
        b_mask_dir = os.path.join(save_dir, 'b_mask')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(ob_dir, exist_ok=True)
        os.makedirs(b_mask_dir, exist_ok=True)
        plt.imsave(os.path.join(input_dir, filename[i]), input_img)
        Image.fromarray(target.astype(np.uint8)).save(os.path.join(target_dir, filename[i]))
        Image.fromarray(ob.astype(np.uint8)).save(os.path.join(ob_dir, filename[i]))
        Image.fromarray(b_mask.astype(np.uint8)).save(os.path.join(b_mask_dir, filename[i]))
def test(model, opt):
    torch.cuda.empty_cache()
    gc.collect()
    print(opt)
    
    test_data = object_prediction_Dataset(opt.data_dir, resize=512, targetresize=True, direction='top', cover_percent=0.1, ) #SegDataset(opt.data_dir, resize=512, targetresize=True) #
    testloader = DataLoader(test_data, 1, shuffle=False)
    device = torch.device('cuda:'+opt.gpu) if opt.gpu != '-1' else torch.device('cpu')
    is_rst = opt.data_dir.split('/')[-1]
    is_rst2 = opt.data_dir.split('/')[-2]
    save_dir = os.path.join(opt.save_dir, f'{model.__class__.__name__}-{is_rst}-{is_rst2}-' + str(len(os.listdir(opt.save_dir))))
    os.makedirs(save_dir)
    
    num_classes = opt.num_classes
    measurement = Measurement(num_classes)
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
    for image, input_img, mask_img, b_mask, filename in tqdm(testloader):#
        image = image.to(device)
        input_img = input_img.to(device)#
        b_mask= b_mask.to(device)
        org_mask = mask_img.clone()
        org_mask = torch.where(org_mask >= 128, 255, 0)
        org_mask = org_mask.cpu().squeeze().numpy()
        mask_img = torch.where(mask_img >= 128, 255, 0)  #####
        mask_cpu = mask_labeling(mask_img.type(torch.long), opt.num_classes)
        with torch.no_grad():
            concat_input = torch.cat((input_img,b_mask),dim=1)
            pred = model(concat_input) #, feature
        pred = F.interpolate(pred, mask_img.shape[-2:], mode='bilinear')
        pred_cpu, mask_cpu = pred.detach().cpu().numpy(), mask_cpu.cpu().numpy()
        acc_pixel, batch_miou, iou_ndarray, precision, recall, f1score = measurement(pred_cpu, mask_cpu)
        ob = pred.clone()
        ob = pred_to_binary_mask(ob)
        ob = ob.cpu().squeeze().numpy()
        b_mask = b_mask.cpu().squeeze().numpy()

        test_acc += acc_pixel
        test_miou += batch_miou
        iou_per_class += iou_ndarray
        
        test_precision += precision
        test_recall += recall
        test_f1score += f1score
            
        if opt.save_img:
            image = F.interpolate(image.detach().cpu(), mask_img.shape[-2:], mode='bilinear')
            save_result_img(image.numpy(), org_mask, pred.detach().cpu().numpy(), ob, b_mask, filename, os.path.join(save_dir, 'imgs'))
    
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
    parser.add_argument('--data_dir', type=str, default='C:/Users/shc01/Downloads/data/cropweed_total/IJRR2017/occ/1/test', help='directory that has data')
    parser.add_argument('--save_dir', type=str, default='C:/Users/shc01/Downloads', help='directory for saving results')
    parser.add_argument('--weights', type=str, default='D:/save/bonirob/mat_test/Unet-ep400-train-1-0/ckpoints/best_miou.pth', help='weights file for test')
    parser.add_argument('--save_img', type=bool, default=True, help='save result images')
    parser.add_argument('--save_txt', type=bool, default=True, help='save training process as txt file')
    parser.add_argument('--show_img', type=bool, default=False, help='show images')
    parser.add_argument('--gpu', type=str, default='0', help='gpu number. -1 is cpu')
    ##############
    parser.add_argument('--model', type=str, default='unet', help='modelname')
    parser.add_argument('--num_classes', type=int, default=2, help='the number of classes')
    
    opt = parser.parse_args()
    assert opt.model in ['unet'], 'opt.model is not available'
    
    if opt.model == 'unet':
        model = models.Unet(num_classes=2, in_channels=4, first_outchannels=64)
    test(model, opt)

   