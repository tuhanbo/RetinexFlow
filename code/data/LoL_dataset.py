import os
import subprocess
import torch.utils.data as data
import numpy as np
import time
import torch
import pickle
import cv2
from torchvision.transforms import ToTensor
import random
import torchvision.transforms as T
from pdb import set_trace as st


class LoL_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.log_low = opt["log_low"] if "log_low" in opt.keys() else False
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.use_noise = opt[
            'noise_prob'] if "noise_prob" in opt.keys() else False  # (opt['noise_prob'] and train) if "noise_prob" in opt.keys() else False
        self.noise_prob = opt['noise_prob'] if self.use_noise else None
        self.noise_level = opt['noise_level'] if "noise_level" in opt.keys() else 0
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        # pdb.set_trace()
        if train:
            self.root = os.path.join(self.root, 'our485')
        else:
            self.root = os.path.join(self.root, 'eval15')
        self.pairs = self.load_pairs(self.root)
        self.to_tensor = ToTensor()
        self.use_crop_edge = opt["use_crop_edge"] if "use_crop_edge" in opt.keys() else False

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):
        low_list = os.listdir(os.path.join(folder_path, 'low'))
        low_list = filter(lambda x: 'png' in x, low_list)
        pairs = []
        for idx, f_name in enumerate(low_list):
            pairs.append(
                [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'low', f_name)), cv2.COLOR_BGR2RGB),  # [:, 4:-4, :],
                 cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'high', f_name)), cv2.COLOR_BGR2RGB),
                 # [:, 4:-4, :],
                 f_name.split('.')[0]])
            # if idx > 10: break
            pairs[-1].append(self.hiseq_color_cv2_img(pairs[-1][0]))
        return pairs

    def hiseq_color_cv2_img(self, img):
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        return result

    def __getitem__(self, item):
        lr, hr, f_name, his = self.pairs[item]

        if self.use_crop:
            hr, lr, his = random_crop(hr, lr, his, self.crop_size)

        if self.center_crop_hr_size:
            hr, lr, his = center_crop(hr, self.center_crop_hr_size), center_crop(lr,
                                                                                 self.center_crop_hr_size), center_crop(
                his, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr, his = random_flip(hr, lr, his)

        if self.use_rot:
            hr, lr, his = random_rotation(hr, lr, his)

        # hr = hr / 255.0
        # lr = lr / 255.0

        # if self.measures is None or np.random.random() < 0.05:
        #     if self.measures is None:
        #         self.measures = {}
        #     self.measures['hr_means'] = np.mean(hr)
        #     self.measures['hr_stds'] = np.std(hr)
        #     self.measures['lr_means'] = np.mean(lr)
        #     self.measures['lr_stds'] = np.std(lr)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        if self.use_crop_edge:
            lr, hr = crop_edge(lr, hr)
        if self.use_noise and random.random() < self.noise_prob:
            lr = torch.randn(lr.shape) * (self.noise_level / 255) + lr
        if self.log_low:
            lr = torch.log(torch.clamp(lr + 1e-3, min=1e-3))
        # if self.gpu:
        #    hr = hr.cuda()
        #    lr = lr.cuda()

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}


class LoL_Dataset_v2(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt
        self.log_low = opt["log_low"] if "log_low" in opt.keys() else False
        self.use_flip = opt["use_flip"] if "use_flip" in opt.keys() else False
        self.use_rot = opt["use_rot"] if "use_rot" in opt.keys() else False
        self.use_crop = opt["use_crop"] if "use_crop" in opt.keys() else False
        self.use_noise = opt[
            'noise_prob'] if "noise_prob" in opt.keys() else False  # (opt['noise_prob'] and train) if "noise_prob" in opt.keys() else False
        self.noise_prob = opt['noise_prob'] if self.use_noise else None
        self.noise_level = opt['noise_level'] if "noise_level" in opt.keys() else 0
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        # pdb.set_trace()
        self.pairs = []
        self.train = train
        for sub_data in ['Synthetic']: # :['Synthetic', 'Real_captured']
            if train:
                root = os.path.join(self.root, sub_data, 'Train')
            else:
                root = os.path.join(self.root, sub_data, 'Test')
            self.pairs.extend(self.load_pairs(root))
        self.to_tensor = ToTensor()
        self.gamma_aug = opt['gamma_aug'] if 'gamma_aug' in opt.keys() else False
        self.use_crop_edge = opt["use_crop_edge"] if "use_crop_edge" in opt.keys() else False

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path):
        low_list = os.listdir(os.path.join(folder_path, 'Low' ))
        low_list = sorted(list(filter(lambda x: 'png' in x, low_list)))
        high_list = os.listdir(os.path.join(folder_path, 'Normal'))
        high_list = sorted(list(filter(lambda x: 'png' in x, high_list)))
        pairs = []
        for idx in range(len(low_list)):
            f_name_low = low_list[idx]
            f_name_high = high_list[idx]
            # if ('r113402d4t' in f_name_low or 'r17217693t' in f_name_low) or self.train: # 'r113402d4t' in f_name_low or 'r116825e2t' in f_name_low or 'r068812d7t' in f_name_low
            pairs.append(
                [cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'Low', f_name_low)),
                                cv2.COLOR_BGR2RGB),  # [:, 4:-4, :],
                    cv2.cvtColor(cv2.imread(os.path.join(folder_path, 'Normal', f_name_high)),
                                cv2.COLOR_BGR2RGB),  # [:, 4:-4, :],
                    f_name_high.split('.')[0]])
            # if idx > 10: break
            pairs[-1].append(self.hiseq_color_cv2_img(pairs[-1][0]))
        return pairs

    def hiseq_color_cv2_img(self, img):
        (b, g, r) = cv2.split(img)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        result = cv2.merge((bH, gH, rH))
        return result

    def __getitem__(self, item):
        lr, hr, f_name, his = self.pairs[item]

        if self.use_crop:
            hr, lr, his = random_crop(hr, lr, his, self.crop_size)

        if self.center_crop_hr_size:
            hr, lr, his = center_crop(hr, self.center_crop_hr_size), center_crop(lr,
                                                                                 self.center_crop_hr_size), center_crop(
                his, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr, his = random_flip(hr, lr, his)

        if self.use_rot:
            hr, lr, his = random_rotation(hr, lr, his)

        if self.gamma_aug:
            gamma = random.uniform(0.4, 2.8)
            lr = gamma_aug(lr, gamma=gamma)
        # hr = hr / 255.0
        # lr = lr / 255.0

        # if self.measures is None or np.random.random() < 0.05:
        #     if self.measures is None:
        #         self.measures = {}
        #     self.measures['hr_means'] = np.mean(hr)
        #     self.measures['hr_stds'] = np.std(hr)
        #     self.measures['lr_means'] = np.mean(lr)
        #     self.measures['lr_stds'] = np.std(lr)

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)
        # if self.use_color_jitter:
        #     lr =
        if self.use_crop_edge:
            lr, hr = crop_edge(lr, hr)
        if self.use_noise and random.random() < self.noise_prob:
            lr = torch.randn(lr.shape) * (self.noise_level / 255) + lr
        if self.log_low:
            lr = torch.log(torch.clamp(lr + 1e-3, min=1e-3))
        # if self.gpu:
        #    hr = hr.cuda()
        #    lr = lr.cuda()
        # if self.concat_histeq:
        #     his = self.to_tensor(his)
        #     lr = torch.cat([lr, his], dim=0)


        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}


def random_flip(img, seg, his_eq):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 1).copy()
    seg = seg if random_choice else np.flip(seg, 1).copy()
    if his_eq is not None:
        his_eq = his_eq if random_choice else np.flip(his_eq, 1).copy()
    return img, seg, his_eq


def gamma_aug(img, gamma=0):
    max_val = img.max()
    img_after_norm = img / max_val
    img_after_norm = np.power(img_after_norm, gamma)
    return img_after_norm * max_val


def random_rotation(img, seg, his):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(0, 1)).copy()
    seg = np.rot90(seg, random_choice, axes=(0, 1)).copy()
    if his is not None:
        his = np.rot90(his, random_choice, axes=(0, 1)).copy()
    return img, seg, his


def random_crop(hr, lr, his_eq, size_hr):
    size_lr = size_hr

    size_lr_x = lr.shape[0]
    size_lr_y = lr.shape[1]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr, :]

    # HR Patch
    start_x_hr = start_x_lr
    start_y_hr = start_y_lr
    hr_patch = hr[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]

    # HisEq Patch
    his_eq_patch = None
    if his_eq is not None:
        his_eq_patch = his_eq[start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr, :]
    return hr_patch, lr_patch, his_eq_patch


def center_crop(img, size):
    if img is None:
        return None
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[border:-border, border:-border, :]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]

def crop_edge(lr, hr):

    lr_processed = crop_if_not_divisible(lr, 16)
    hr_processed = crop_if_not_divisible(hr, 16)

    # Return processed variables along with unchanged f_name
    return lr_processed, hr_processed

def crop_if_not_divisible(data, divisor):
    """
    Crop the edges of a numpy array if its dimensions are not divisible by a given divisor.

    Parameters:
    - data: Input numpy array to be cropped.
    - divisor: Divisor for dimensions (e.g., 16).

    Returns:
    - Cropped numpy array.
    """

    # Get current dimensions
    channels, height, width = data.size()

    # Calculate new dimensions that are divisible by the divisor
    new_height = height - (height % divisor)
    new_width = width - (width % divisor)

    # Crop the edges if necessary
    if new_height < height or new_width < width:
        start_y = (height - new_height) // 2
        end_y = start_y + new_height
        start_x = (width - new_width) // 2
        end_x = start_x + new_width
        cropped_data = data[:, start_y:end_y, start_x:end_x]
    else:
        cropped_data = data  # No cropping needed

    return cropped_data


from pathlib import Path
from tqdm import tqdm
from torchvision import transforms


class RELLISUR_Dataset(data.Dataset):
    def __init__(self, opt, train, all_opt):
        self.root = opt["root"]
        self.opt = opt

        self.log_low = opt.get("log_low", False)
        self.use_flip = opt.get("use_flip", False)
        self.use_rot = opt.get("use_rot", False)
        self.use_crop = opt.get("use_crop", False)
        self.use_noise = opt.get("noise_prob", False)
        self.noise_prob = opt.get("noise_prob", 0.0) if self.use_noise else None
        self.noise_level = opt.get("noise_level", 0)
        self.center_crop_hr_size = opt.get("center_crop_hr_size", None)
        self.crop_size = opt.get("GT_size", None)
        self.scale = opt.get("scale", None)

        # 选择训练集或测试集
        if train:
            self.root = os.path.join(self.root, 'Train')
        else:
            self.root = os.path.join(self.root, 'Test_crop')

        # 预加载数据对
        self.pairs = self.load_pairs(self.root, train)
        self.to_tensor = transforms.ToTensor()
        self.use_crop_edge = opt["use_crop_edge"] if "use_crop_edge" in opt.keys() else False

    def __len__(self):
        return len(self.pairs)

    def load_pairs(self, folder_path, train):
        """加载 LQ 和 GT 图像对"""
        folder_path = Path(folder_path)
        if train:
            high_folder = folder_path / "NLHR-Duplicates/X1"
            low_folder = folder_path / "LLLR"
        else:
            high_folder = folder_path / "NLHR-Duplicates/X1"
            low_folder = folder_path / "LLLR"

        # 获取所有低分辨率图像的文件名
        low_files = sorted(low_folder.glob("*.png"))

        pairs = []
        for low_path in tqdm(low_files, desc="Loading dataset", ncols=80):
            f_name = low_path.name
            high_path = high_folder / f_name

            # 读取图像
            low_img = cv2.imread(str(low_path))
            high_img = cv2.imread(str(high_path))

            # 转换为 RGB 格式
            low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
            high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

            pairs.append([low_img, high_img, f_name.split('.')[0]])

        return pairs

    def __getitem__(self, idx):
        lr, hr, f_name = self.pairs[idx]

        if self.use_crop:
            hr, lr = self.random_crop(hr, lr,  self.crop_size, self.scale)

        if self.center_crop_hr_size:
            hr, lr = self.center_crop(hr, self.center_crop_hr_size), self.center_crop(lr, self.center_crop_hr_size)

        if self.use_flip:
            hr, lr = self.random_flip(hr, lr)

        if self.use_rot:
            hr, lr = self.random_rotation(hr, lr)
        # 转 Tensor
        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        if self.use_crop_edge:
            lr, hr = crop_edge(lr, hr)

        # 添加噪声
        if self.use_noise and random.random() < self.noise_prob:
            lr = torch.randn(lr.shape) * (self.noise_level / 255) + lr

        # 低光对数变换
        if self.log_low:
            lr = torch.log(torch.clamp(lr + 1e-3, min=1e-3))

        return {'LQ': lr, 'GT': hr, 'LQ_path': f_name, 'GT_path': f_name}

    def random_crop(self, hr, lr, size_hr, scale=1):
        """随机裁剪 HR, LR 图像"""
        size_lr = size_hr // scale
        h_lr, w_lr = lr.shape[:2]
        start_x_lr = np.random.randint(0, max(1, h_lr - size_lr + 1))
        start_y_lr = np.random.randint(0, max(1, w_lr - size_lr + 1))

        # 获取 LR 和 HR patch
        lr_patch = lr[start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr]
        hr_patch = hr[start_x_lr * scale: start_x_lr * scale + size_hr,
                   start_y_lr * scale: start_y_lr * scale + size_hr]

        return hr_patch, lr_patch

    def random_flip(self, img, seg):
        """随机翻转图像"""
        random_choice = np.random.choice([True, False])
        img = img if random_choice else np.flip(img, 1).copy()
        seg = seg if random_choice else np.flip(seg, 1).copy()
        return img, seg

    def random_rotation(self, img, seg):
        """随机旋转图像"""
        random_choice = np.random.choice([0, 1, 3])
        img = np.rot90(img, random_choice, axes=(0, 1)).copy()
        seg = np.rot90(seg, random_choice, axes=(0, 1)).copy()
        return img, seg

    def center_crop(self, img, size):
        """中心裁剪"""
        h, w = img.shape[:2]
        top = (h - size) // 2
        left = (w - size) // 2
        return img[top:top + size, left:left + size]

    def center_crop_tensor(self, img, size):
        assert img.shape[2] == img.shape[3], img.shape
        border_double = img.shape[2] - size
        assert border_double % 2 == 0, (img.shape, size)
        border = border_double // 2
        return img[:, :, border:-border, border:-border]
