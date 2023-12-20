import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class SegmentationDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_filenames = []  # List to store image file names
        self.mask_filenames = []  # List to store mask file names

        # Load image and mask file names
        self.images_dir = os.path.join(self.data_path, 'images_png')
        self.masks_dir = os.path.join(self.data_path, 'masks_png')
        self.image_filenames = sorted(os.listdir(self.images_dir))
        self.mask_filenames = sorted(os.listdir(self.masks_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')
        if self.transform is not None:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)
            
        imageT = T.ToTensor()
        maskT = T.PILToTensor()

        #image = np.array(image)
        #mask = np.array(mask)
        #image = np.transpose(image, (1, 2, 0)).astype(np.float32)
        #return (torch.from_numpy(image), torch.from_numpy(mask))
        return imageT(image), maskT(mask)