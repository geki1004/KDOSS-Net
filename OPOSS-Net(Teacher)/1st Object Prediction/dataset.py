import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.ops import masks_to_boxes

class SegDataset(Dataset):
    def __init__(self, data_dir, resize=512, inputresize=True, targetresize=False, transform=None, target_transform=None, randomaug=None):
        self.img_dir = os.path.join(data_dir, 'input')
        self.mask_dir = os.path.join(data_dir, 'target')
        self.resize = resize
        self.inputresize = inputresize
        self.targetresize = targetresize
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.randomaug = randomaug
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.inputresize: image = image.resize((self.resize, self.resize), resample=Image.BILINEAR)
        image = TF.to_tensor(image)

        mask_path = os.path.join(self.mask_dir, filename)
        mask = Image.open(mask_path).convert('L')  # size : (W, H), grayscale image
        if self.targetresize: mask = mask.resize((self.resize, self.resize), resample=Image.NEAREST)
        mask = np.array(mask)  # (H, W)
        mask = torch.from_numpy(mask)
        mask = mask.to(torch.int64)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        if self.randomaug:
            # Data Augmentations
            aug = random.randint(0, 7)
            if aug == 1:
                image = image.flip(1)
                mask = mask.flip(0)
            elif aug == 2:
                image = image.flip(2)
                mask = mask.flip(1)
            elif aug == 3:
                image = torch.rot90(image, dims=(1, 2))
                mask = torch.rot90(mask, dims=(0, 1))
            elif aug == 4:
                image = torch.rot90(image, dims=(1, 2), k=2)
                mask = torch.rot90(mask, dims=(0, 1), k=2)
            elif aug == 5:
                image = torch.rot90(image, dims=(1, 2), k=-1)
                mask = torch.rot90(mask, dims=(0, 1), k=-1)
            elif aug == 6:
                image = torch.rot90(image.flip(1), dims=(1, 2))
                mask = torch.rot90(mask.flip(0), dims=(0, 1))
            elif aug == 7:
                image = torch.rot90(image.flip(2), dims=(1, 2))
                mask = torch.rot90(mask.flip(1), dims=(0, 1))
        
        return image, mask, filename

class object_prediction_Dataset(Dataset):
    def __init__(self, data_dir, resize=512, inputresize=True, targetresize=True, transform=None, target_transform=None, direction='top', cover_percent=0.1, randomaug=None): #초기화
        self.img_dir = os.path.join(data_dir, 'input')
        self.mask_dir = os.path.join(data_dir, 'target')
        self.resize = resize
        self.inputresize = inputresize
        self.targetresize = targetresize
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.direction = direction
        self.cover_percent = cover_percent
        self.randomaug = randomaug

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx): #인덱싱
        filename = self.images[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.inputresize: image = image.resize((self.resize, self.resize), resample=Image.BILINEAR)
        width, height = image.size
        image = TF.to_tensor(image)

        mask_path = os.path.join(self.mask_dir, filename)
        mask = Image.open(mask_path).convert('L')  # size : (W, H), grayscale image
        if self.targetresize: mask = mask.resize((self.resize, self.resize), resample=Image.NEAREST)
        mask = np.array(mask)  # (H, W)
        mask = torch.from_numpy(mask)
        mask = mask.to(torch.int64)
        if self.randomaug:
            # Data Augmentations
            aug = random.randint(0, 7)
            if aug == 1:
                image = image.flip(1)
                mask = mask.flip(0)
            elif aug == 2:
                image = image.flip(2)
                mask = mask.flip(1)
            elif aug == 3:
                image = torch.rot90(image, dims=(1, 2))
                mask = torch.rot90(mask, dims=(0, 1))
            elif aug == 4:
                image = torch.rot90(image, dims=(1, 2), k=2)
                mask = torch.rot90(mask, dims=(0, 1), k=2)
            elif aug == 5:
                image = torch.rot90(image, dims=(1, 2), k=-1)
                mask = torch.rot90(mask, dims=(0, 1), k=-1)
            elif aug == 6:
                image = torch.rot90(image.flip(1), dims=(1, 2))
                mask = torch.rot90(mask.flip(0), dims=(0, 1))
            elif aug == 7:
                image = torch.rot90(image.flip(2), dims=(1, 2))
                mask = torch.rot90(mask.flip(1), dims=(0, 1))
        #-------------------------------------------------
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        boxesizeidx = boxes.size(0)
        if boxesizeidx != 2:
            roi_list = boxes[0].int()
        else:
            min_left = torch.min(boxes[0, 0:2], boxes[1, 0:2])
            max_right = torch.max(boxes[0, 2:4], boxes[1, 2:4])
            result_tensor = torch.cat([min_left, max_right], dim=0)
            roi_list = result_tensor.int()

        roi_left, roi_top, roi_right, roi_bottom = roi_list
        roi_width = roi_right - roi_left
        roi_height = roi_bottom - roi_top

        b_mask = np.ones((1, height, width), dtype=np.uint8)

        if self.direction == 'top':
            b_mask[:, :roi_top+int(roi_height*self.cover_percent),:] = 0
        # 하단
        elif self.direction == 'bottom':
            b_mask[:,roi_bottom-int(roi_height*self.cover_percent):height,:] = 0
        # 좌측
        elif self.direction == 'left':
            b_mask[:, :, :roi_left+int(roi_width*self.cover_percent)] = 0
        # 우측
        elif self.direction == 'right':
            b_mask[:, :, roi_right-int(roi_width*self.cover_percent):width] = 0

        blind_image = image * b_mask

        b_mask =torch.tensor(b_mask).float()

        if self.transform:
            blind_image = self.transform(blind_image)
            image = self.transform(image)
            b_mask = self.transform(b_mask)

        if self.target_transform:
            mask = self.target_transform(mask)


        return image, blind_image.float(), mask, b_mask, filename