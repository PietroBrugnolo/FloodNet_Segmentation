# -------------------------------------------------------------------------
# Author:   Pietro Brugnolo
# Date:     20/11/2023
# Version:  1
# -------------------------------------------------------------------------

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class FloodNetDataset(Dataset):
    """
            classes:
            
            "background",
            "building-flooded",
            "building non-flooded",
            "road flooded",
            "road non-flooded",
            "water",
            "tree",
            "vehicle",
            "pool",
            "grass",
            
    """

    def __init__(
        self, base_folder="/kaggle/input/floodnet-labeled/segmentation kaggle/Train", transform=lambda x: x,  target_transform=lambda y: y
    ) -> None:
        super().__init__()
        self.base_folder = base_folder
        self.im_files = [f for f in os.listdir(self.base_folder) if f.endswith(".jpg")]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        img_file = os.path.join(self.base_folder, self.im_files[index])
        gt_file = img_file.replace(".jpg", "_lab.png")
        img = Image.open(img_file)
        label = Image.open(gt_file)
        state = torch.get_rng_state()
        img = self.transform(img)
        torch.set_rng_state(state)
        label = self.target_transform(label)

        return img, label