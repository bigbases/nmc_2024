import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd 
import os 
import numpy as np 
class ODIRDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Fundus']
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label_vector = self.dataframe.loc[idx][6:-1].values
        label_vector = label_vector.astype(float)

        label_tensor = torch.tensor(label_vector)
        # label_vector = np.array(self.dataframe.iloc[idx]['diagnosis'], dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

