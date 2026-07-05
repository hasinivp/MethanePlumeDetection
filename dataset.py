import torch
import os 
import numpy as np
from torch.utils.data import Dataset

#how to load one image mask pair
class MethaneDataset(Dataset):
    def __init__(self, image_dir, mask_dir): #self is instance of class 
        self.image_dir = image_dir 
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(self.image_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_file = self.images[idx]
        mask_file = image_file.replace("image_", "mask_") 

        image = np.load(os.path.join(self.image_dir, image_file)).astype(np.float32)
        mask = np.load(os.path.join(self.mask_dir, mask_file)).astype(np.float32)

        image = torch.tensor(image).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        return image, mask