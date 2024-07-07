import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from PIL import Image
import pandas as pd 
import os 
import numpy as np 
class ODIRDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        print(image_dir)
        # /data/public_data/cropped_image/train_images
        self.CLASSES = ['N','D','G','C','A','H','M','O']
        data = image_dir.split('/')[-1]
        if 'train_images' == data: 
            # /data_2/national_AI_DD/public_data/cropped_image/cropped_train.csv
            df_path = image_dir.replace('train_images','new_ODIR_train.csv')
            self.dataframe = pd.read_csv(df_path)
            pass
        elif 'test_images' == data:
            df_path = image_dir.replace('test_images','new_ODIR_test.csv')
            self.dataframe = pd.read_csv(df_path)
            pass
        
        elif 'val_images' == data:
            df_path = image_dir.replace('val_images','new_ODIR_valid.csv')
            self.dataframe = pd.read_csv(df_path)
            pass
        else:
            print('wrong')
        # print(self.dataframe)
        self.image_dir = image_dir
        self.transform = transform
        self.n_classes = len(self.CLASSES)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Fundus']
        img_path = os.path.join(self.image_dir, img_name)
        image = io.read_image(img_path)
        label_vector = self.dataframe.loc[idx][6:-1].values
        label_vector = label_vector.astype(float)

        label_tensor = torch.tensor(label_vector)
        # label_vector = np.array(self.dataframe.iloc[idx]['diagnosis'], dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor

