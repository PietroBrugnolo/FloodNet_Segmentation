# -------------------------------------------------------------------------
# Author:   Pietro Brugnolo
# Date:     20/11/2023
# Version:  1
# -------------------------------------------------------------------------

import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class FloodNetDataset_pseudolabels(Dataset):
    def __init__(self, base_folder, weak_transform=lambda x: x, strong_transform=lambda y: y) -> None:
        super().__init__()
        self.base_folder = base_folder
        self.im_files = [f for f in os.listdir(self.base_folder) if f.endswith(".jpg")]
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        img_file = os.path.join(self.base_folder, self.im_files[index])
        img = Image.open(img_file)
        state = torch.get_rng_state()
        img_aug = self.strong_transform(img)
        torch.set_rng_state(state)
        img_target = self.weak_transform(img)
        return img_aug, img_target