"""
@time: 2025/01/08
@file: my_dataset.py
@author: WD                     ___       __   ________            
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ / 
                                ____/|__/      /_____/  

自定义数据集
"""

from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import os


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            # self.image_root = os.path.join(root, "DUTS-TR", "DUTS-TR-Image")
            # self.mask_root = os.path.join(root, "DUTS-TR", "DUTS-TR-Mask")
            self.image_root = os.path.join(root, "segimages", "train")
            self.mask_root = os.path.join(root, "seglabels", "train")
        else:
            # self.image_root = os.path.join(root, "DUTS-TE", "DUTS-TE-Image")
            # self.mask_root = os.path.join(root, "DUTS-TE", "DUTS-TE-Mask")
            self.image_root = os.path.join(root, "segimages", "val")
            self.mask_root = os.path.join(root, "seglabels", "val")
        assert os.path.exists(self.image_root), f"❌ path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"❌ path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
        # mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".jpg")]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."
        if train:
            print('✅ Load {} train images success!'.format(len(image_names)))
        else:
            print('✅ Load {} Val images success!'.format(len(image_names)))
        # check images and mask
        re_mask_names = []
        for p in image_names:
            # mask_name = p.replace(".jpg", ".png")
            mask_name = p.replace(".jpg", ".jpg")
            assert mask_name in mask_names, f"❌ {p} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        assert image is not None, f"❌ failed to read image: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = image.shape

        target = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert target is not None, f"❌ failed to read mask: {mask_path}"

        if self.transforms is not None:
            # image, target = self.transforms(image, target)
            image = self.transforms(image)
            target = self.transforms(target)

        return image, target

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
