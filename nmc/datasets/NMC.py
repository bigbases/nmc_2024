import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import os 
import numpy as np 
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.io import read_image
from itertools import chain




class NMCDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        print(image_dir)
        data = image_dir.split('/')[-1]
        if 'train_images' == data: 
            df_path = image_dir.replace('train_images', 'nmc_train.csv')
        elif 'test_images' == data:
            df_path = image_dir.replace('test_images', 'nmc_test.csv')
        elif 'val_images' == data:
            df_path = image_dir.replace('val_images', 'nmc_valid.csv')
        else:
            raise ValueError('Invalid image directory name.')
        
        self.dataframe = pd.read_csv(df_path)
        self.dataframe = self.dataframe.dropna()
        
        # Ensure the label column is treated as a list of labels
        def process_label(x):
            if isinstance(x, str):
                return x.split(',')
            else:
                raise ValueError(f"Unexpected label value: {x}")
        
        self.dataframe['label'] = self.dataframe['label'].apply(process_label)
        
        # 필터링하여 0~10 범위의 레이블만 남기기
        def filter_labels(labels):
            return [int(label) for label in labels if label and 0 <= int(label) <= 10]
        
        self.dataframe['label'] = self.dataframe['label'].apply(filter_labels)
        
        # Initialize the MultiLabelBinarizer and fit_transform the label column
        self.mlb = MultiLabelBinarizer(classes=[i for i in range(11)])
        self.one_hot_labels = self.mlb.fit_transform(self.dataframe['label'])
        
        # print("Classes found by MultiLabelBinarizer:", self.mlb.classes_)
        # print("One-hot encoded labels shape:", self.one_hot_labels.shape)

        # 데이터프레임의 고유한 레이블 값 확인
        unique_labels = set(label for sublist in self.dataframe['label'] for label in sublist)
        print("Unique labels in dataframe:", unique_labels)
        self.CLASSES = [0,1,2,3,4,5,6,7,8,9,10]
        self.n_classes = len(self.CLASSES)
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image']
        img_path = os.path.join(self.image_dir, img_name)
        image = read_image(img_path)
        
        label_vector = self.one_hot_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_vector, dtype=torch.float32)

class EpisodicNMCDataset(Dataset):
    def __init__(self, image_dir, n_way, k_shot, q_query, transform=None):
        super().__init__()
        data = image_dir.split('/')[-1]
        if 'train_images' == data: 
            df_path = image_dir.replace('train_images', 'nmc_train.csv')
        elif 'test_images' == data:
            df_path = image_dir.replace('test_images', 'nmc_test.csv')
        elif 'val_images' == data:
            df_path = image_dir.replace('val_images', 'nmc_valid.csv')
        else:
            raise ValueError('Invalid image directory name.')
        
        self.dataframe = pd.read_csv(df_path)
        self.dataframe = self.dataframe.dropna()
        
        def process_label(x):
            if isinstance(x, str):
                return x.split(',')
            else:
                raise ValueError(f"Unexpected label value: {x}")
        self.dataframe['label'] = self.dataframe['label'].apply(process_label)
        
        def filter_labels(labels):
            return [int(label) for label in labels if label and 0 <= int(label) <= 10]
        
        self.dataframe['label'] = self.dataframe['label'].apply(filter_labels)
        
        # Initialize the MultiLabelBinarizer and fit_transform the label column
        self.mlb = MultiLabelBinarizer(classes=[int(i) for i in range(11)])
        self.one_hot_labels = self.mlb.fit_transform(self.dataframe['label'])
        self.unique_labels = set(label for sublist in self.dataframe['label'] for label in sublist)
        print("Unique labels in dataframe:", self.unique_labels)
        self.image_dir = image_dir
        self.transform = transform

        self.CLASSES = [0,1,2,3,4,5,6,7,8,9,10]
        self.n_classes = len(self.CLASSES)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        
        self.dataframe = self.dataframe.reset_index(drop=True)
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.dataframe):
            raise IndexError(f"Index {idx} is out-of-bounds")
        img_name = self.dataframe.iloc[idx]['image']
        img_path = os.path.join(self.image_dir, img_name)
        image = read_image(img_path)
        
        label_vector = self.one_hot_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_vector, dtype=torch.float32)

    def create_episode(self):
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        
        chosen_classes = np.random.choice(list(self.unique_labels), self.n_way, replace=False)
        chosen_classes_set = set(chosen_classes)
        # print(f"chosen_classes_set: {chosen_classes_set}")

        # Find all indices where the label contains at least one of the chosen classes
        cls_indices = self.dataframe[self.dataframe['label'].apply(lambda labels: bool(set(labels) & chosen_classes_set))].index.tolist()
        
        # 클래스 조합으로 이루어진 DF 생성
        cls_indices_df = self.dataframe.iloc[cls_indices]
        # print(f"cls_indices (length: {len(cls_indices)}): {cls_indices}")
        
        if len(cls_indices) < self.k_shot + self.q_query:
            raise ValueError("Not enough samples to create an episode with the chosen classes.")
        
        # 개별 고유 클래스에 대해 k_shot 만큼 support_indices에 추가
        support_indices = []
        for i in chosen_classes:    
            temp_df = cls_indices_df[cls_indices_df['label'].apply(lambda x: i in x)]
            selected_rows = temp_df.sample(n=self.k_shot)
            selected_indices = selected_rows.index.tolist() 
            support_indices += selected_indices  
        
        support_indices = list(set(support_indices))
        # print(f"support_indices: {support_indices}")

        # q_query만큼 클래스 조합으로 이루어진 DF에서 추출
        query_indices = np.random.choice(cls_indices, self.q_query, replace=False)
        # print(f"query_indices: {query_indices}")
        
        for idx in support_indices:
            if idx < 0 or idx >= len(self.dataframe):
                raise IndexError(f"Support index {idx} is out-of-bounds")
            img, label = self.__getitem__(idx)
            support_x.append(img)
            support_y.append(label)

        for idx in query_indices:
            if idx < 0 or idx >= len(self.dataframe):
                raise IndexError(f"Query index {idx} is out-of-bounds")
            img, label = self.__getitem__(idx)
            query_x.append(img)
            query_y.append(label)

        return torch.stack(support_x), torch.stack(support_y), torch.stack(query_x), torch.stack(query_y)

# Epdisode를 만들어주는 DataLoader 추가
def episodic_dataloader(dataset, num_episodes):
    for _ in range(num_episodes):
        yield dataset.create_episode()
        
        