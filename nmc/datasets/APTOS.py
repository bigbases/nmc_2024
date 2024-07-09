import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io
import pandas as pd 
import os 
import numpy as np 
class APTOSDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        print(image_dir)
        # /data/public_data/cropped_image/train_images
        self.CLASSES = ['Norma','Mild','Moderate Disease Level','Server','Proliferative']
        data = image_dir.split('/')[-1]
        # print(data)
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
        self.n_classes = len(self.CLASSES)

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['id_code']
        img_name +='.png'
        img_path = os.path.join(self.image_dir, img_name)
        #image = Image.open(img_path).convert('RGB')
        image = io.read_image(img_path)
        one_hot = np.zeros(5)
        one_hot[self.dataframe.iloc[idx]['diagnosis']] = 1
        # label_vector = np.array(self.dataframe.iloc[idx]['diagnosis'], dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(one_hot)
    

class EpisodicAPTOSDataset(Dataset):
    def __init__(self, dataframe, image_dir, n_way, k_shot, q_query, transform=None):
        super().__init__()
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['id_code'] + '.png'
        img_path = os.path.join(self.image_dir, img_name)
        #image = Image.open(img_path).convert('RGB')
        image = io.read_image(img_path)
        
        one_hot = np.zeros(5)
        one_hot[self.dataframe.iloc[idx]['diagnosis']] = 1

        if self.transform:
            image = self.transform(image)

        return image, label

    def create_episode(self):
        support_x = []
        support_y = []
        query_x = []
        query_y = []

        chosen_classes = np.random.choice(self.dataframe['diagnosis'].unique(), self.n_way, replace=False)

        for cls in chosen_classes:
            cls_indices = self.dataframe[self.dataframe['diagnosis'] == cls].index.tolist()
            chosen_indices = np.random.choice(cls_indices, self.k_shot + self.q_query, replace=False)
            support_indices = chosen_indices[:self.k_shot]
            query_indices = chosen_indices[self.k_shot:]

            for idx in support_indices:
                img, label = self.__getitem__(idx)
                support_x.append(img)
                support_y.append(label)

            for idx in query_indices:
                img, label = self.__getitem__(idx)
                query_x.append(img)
                query_y.append(label)

        return torch.stack(support_x), torch.tensor(support_y), torch.stack(query_x), torch.tensor(query_y)
    


# Epdisode를 만들어주는 DataLoader 추가
def episodic_dataloader(dataset, num_episodes):
    for _ in range(num_episodes):
        yield dataset.create_episode()
