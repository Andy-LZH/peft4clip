import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

classes_to_id = {'Arborio': 0, 'Basmati': 1, 'Ipsala': 2, 'Jasmine': 3, 'Karacadag': 4}

class Rice_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.rice_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    
    def __len__(self):
        return len(self.rice_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.rice_frame.iloc[idx, 1])
        image = Image.open(img_name)
        label = self.rice_frame.iloc[idx, 2]
        label_id = classes_to_id[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_id, idx
