import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.ops import masks_to_boxes

class image_outpainting_Dataset(Dataset):
    def __init__(self, data_dir, resize=512, inputresize=True, targetresize=True, transform=None, target_transform=None, direction='top', cover_percent=0.2, randomaug=None): #초기화
        self.img_dir = os.path.join(data_dir, 'input')
        self.mask_dir = os.path.join(data_dir, 'target')
        self.ob_dir = os.path.join(data_dir, 'ob')
        self.resize = resize
        self.inputresize = inputresize
        self.targetresize = targetresize
        self.images = os.listdir(self.img_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.direction = direction
        self.cover_percent = cover_percent
        self.randomaug = randomaug

        self.image_groups = self._group_files(self.img_dir)
        self.mask_groups = self._group_files(self.mask_dir)
        self.ob_groups = self._group_files(self.ob_dir)

        self.group_names = list(self.image_groups.keys())

    def _group_files(self, directory):
        """디렉토리 내 파일을 그룹화."""
        files = os.listdir(directory)
        grouped_files = {}

        for file_name in files:
            # 파일 이름에서 숫자 부분을 처리하고, 없으면 _0을 추가
            if file_name.count('_') == 1:  # '_0'이 없는 파일 (예: rgb_00023.png)
                base_name = file_name.split('.')[0]
            else:
                base_name = '_'.join(file_name.split('_')[:2])  # 'rgb_00023'

            # 그룹 이름이 아직 grouped_files에 없다면 새로운 리스트를 생성합니다.
            if base_name not in grouped_files:
                grouped_files[base_name] = []

            # 해당 그룹 이름에 파일 경로를 추가합니다.
            grouped_files[base_name].append(os.path.join(directory, file_name))

        return grouped_files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx): #인덱싱
        group_name = self.group_names[idx]
        idx = random.choice(range(len(self.image_groups[group_name])))  # 그룹의 길이 범위에서 랜덤 인덱스 선택

        img_path = self.image_groups[group_name][idx]
        mask_path = self.mask_groups[group_name][idx]
        ob_path = self.ob_groups[group_name][idx]


        image = Image.open(img_path).convert('RGB')
        if self.inputresize: image = image.resize((self.resize, self.resize), resample=Image.BILINEAR)
        width, height = image.size
        image = TF.to_tensor(image)

        mask = Image.open(mask_path).convert('L')  # size : (W, H), grayscale image
        if self.targetresize: mask = mask.resize((self.resize, self.resize), resample=Image.NEAREST)

        mask = np.array(mask)  # (H, W)
        mask = torch.from_numpy(mask)
        mask = mask.to(torch.int64)

        ob = Image.open(ob_path).convert('L')  # size : (W, H), grayscale image
        if self.targetresize: ob = ob.resize((self.resize, self.resize), resample=Image.NEAREST)
        ob = np.array(ob).astype(np.uint8)
        ob = np.expand_dims(ob, 0)
        ob = torch.tensor(ob).float()

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
        #----------------------------------------------------

        #
        b_mask =torch.tensor(b_mask).float()

        if self.transform:
            blind_image = self.transform(blind_image)
            image = self.transform(image)
            ob = self.transform(ob)

        if self.target_transform:
            mask = self.target_transform(mask)


        return image, blind_image.float(), mask, b_mask, ob, os.path.basename(img_path)

class blind_SegDataset_aug(Dataset):
    def __init__(self, data_dir, resize=512, inputresize=True, targetresize=True, transform=None, target_transform=None, direction='top', cover_percent=0.2, randomaug=True, aug=0): #초기화
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
        self.aug = aug

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):  # 인덱싱
        filename = self.images[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.inputresize:
            image = image.resize((self.resize, self.resize), resample=Image.BILINEAR)
        width, height = image.size
        image = TF.to_tensor(image)

        mask_path = os.path.join(self.mask_dir, filename)
        mask = Image.open(mask_path).convert('L')  # size : (W, H), grayscale image
        if self.targetresize:
            mask = mask.resize((self.resize, self.resize), resample=Image.NEAREST)
        mask1 = mask
        mask = np.array(mask)  # (H, W)
        mask = torch.from_numpy(mask)
        mask = mask.to(torch.int64)

        outputs = []

        if self.randomaug:
            aug = self.aug
            aug_image = image.clone()
            aug_mask = mask.clone()

            if aug == 1:
                aug_image = aug_image.flip(1)
                aug_mask = aug_mask.flip(0)
            elif aug == 2:
                aug_image = aug_image.flip(2)
                aug_mask = aug_mask.flip(1)
            elif aug == 3:
                aug_image = torch.rot90(aug_image, dims=(1, 2))
                aug_mask = torch.rot90(aug_mask, dims=(0, 1))
            elif aug == 4:
                aug_image = torch.rot90(aug_image, dims=(1, 2), k=2)
                aug_mask = torch.rot90(aug_mask, dims=(0, 1), k=2)
            elif aug == 5:
                aug_image = torch.rot90(aug_image, dims=(1, 2), k=-1)
                aug_mask = torch.rot90(aug_mask, dims=(0, 1), k=-1)
            elif aug == 6:
                aug_image = torch.rot90(aug_image.flip(1), dims=(1, 2))
                aug_mask = torch.rot90(aug_mask.flip(0), dims=(0, 1))
            elif aug == 7:
                aug_image = torch.rot90(aug_image.flip(2), dims=(1, 2))
                aug_mask = torch.rot90(aug_mask.flip(1), dims=(0, 1))


            obj_ids = torch.unique(aug_mask)
            obj_ids = obj_ids[1:]
            masks = aug_mask == obj_ids[:, None, None]
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
                b_mask[:, :roi_top + int(roi_height * self.cover_percent), :] = 0
            elif self.direction == 'bottom':
                b_mask[:, roi_bottom - int(roi_height * self.cover_percent):height, :] = 0
            elif self.direction == 'left':
                b_mask[:, :, :roi_left + int(roi_width * self.cover_percent)] = 0
            elif self.direction == 'right':
                b_mask[:, :, roi_right - int(roi_width * self.cover_percent):width] = 0

            blind_image = aug_image * b_mask  # + (1-b_mask)*0.1
            b_mask = torch.tensor(b_mask).float()

            if self.transform:
                blind_image = self.transform(blind_image)
                aug_image = self.transform(aug_image)

            if self.target_transform:
                aug_mask = self.target_transform(aug_mask)

            filename = os.path.splitext(filename)[0]

            aug_filename = f'{filename}_{aug}.png'  # 파일 이름 수정

        return aug_image, blind_image.float(), aug_mask, b_mask, aug_filename

