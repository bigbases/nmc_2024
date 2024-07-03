import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd 
import os 
import numpy as np 
class APTOSDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        print(image_dir)
        # /data/public_data/cropped_image/train_images
        data = image_dir.split('/')[-1]
        if 'train_images' == data: 
            # /data_2/national_AI_DD/public_data/cropped_image/cropped_train.csv
            df_path = image_dir.replace('train_images','cropped_train.csv')
            self.dataframe = pd.read_csv(df_path)
            pass
        elif 'test_images' == data:
            df_path = image_dir.replace('test_images','cropped_test.csv')
            self.dataframe = pd.read_csv(df_path)
            pass
        
        elif 'val_images' == data:
            df_path = image_dir.replace('val_images','cropped_valid.csv')
            self.dataframe = pd.read_csv(df_path)
            pass
        else:
            print('wrong')
            
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['id_code']
        img_name +='.png'
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        label_vector = np.array(self.dataframe.iloc[idx]['diagnosis'], dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_vector)

