import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import os 
import numpy as np 
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.io import read_image
from itertools import chain
import sys

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

# class EpisodicNMCDataset(Dataset):
#     def __init__(self, image_dir, n_way, k_shot, q_query, minor_cls=[4, 5], transform=None):
#         super().__init__()
#         data = image_dir.split('/')[-1]
#         if 'train_images' == data: 
#             df_path = image_dir.replace('train_images', 'nmc_train.csv')
#             self.is_train = True
#         elif 'test_images' == data:
#             df_path = image_dir.replace('test_images', 'nmc_test.csv')
#             self.is_train = False
#         elif 'val_images' == data:
#             df_path = image_dir.replace('val_images', 'nmc_valid.csv')
#             self.is_train = False
#         else:
#             raise ValueError('Invalid image directory name.')
        
#         self.dataframe = pd.read_csv(df_path)
#         self.dataframe = self.dataframe.dropna()
        
#         def process_label(x):
#             if isinstance(x, str):
#                 return x.split(',')
#             else:
#                 raise ValueError(f"Unexpected label value: {x}")
            
#         self.dataframe['label'] = self.dataframe['label'].apply(process_label)
        
#         def filter_labels(labels):
#             return [int(label) for label in labels if label and 0 <= int(label) <= 10]
        
#         self.dataframe['label'] = self.dataframe['label'].apply(filter_labels)
        
#         self.minor_cls = set(minor_cls)
#         if self.is_train:
#             self.dataframe = self.dataframe[
#                 ~self.dataframe['label'].apply(lambda labels: bool(self.minor_cls & set(labels)))
#             ]

#         self.mlb = MultiLabelBinarizer(classes=[int(i) for i in range(11)])
#         self.one_hot_labels = self.mlb.fit_transform(self.dataframe['label'])
#         self.unique_labels = set(label for sublist in self.dataframe['label'] for label in sublist)
        
#         print("Unique labels in dataframe:", self.unique_labels)
#         print(len(self.dataframe))
#         self.image_dir = image_dir
#         self.transform = transform

#         self.CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         self.n_classes = len(self.CLASSES)
#         self.n_way = n_way
#         self.k_shot = k_shot
#         self.q_query = q_query
        
#         self.dataframe = self.dataframe.reset_index(drop=True)
    
#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         if idx < 0 or idx >= len(self.dataframe):
#             raise IndexError(f"Index {idx} is out-of-bounds")
#         img_name = self.dataframe.iloc[idx]['image']
#         img_path = os.path.join(self.image_dir, img_name)
#         image = read_image(img_path)
        
#         label_vector = self.one_hot_labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, torch.tensor(label_vector, dtype=torch.float32)

#     def create_episode(self):
#         support_x = []
#         support_y = []
#         query_x = []
#         query_y = []
        
#         chosen_classes = np.random.choice(list(self.unique_labels), self.n_way, replace=False)
#         chosen_classes_set = set(chosen_classes)
#         # print(f"chosen_classes_set: {chosen_classes_set}")

#         # Find all indices where the label contains at least one of the chosen classes
#         cls_indices = self.dataframe[self.dataframe['label'].apply(lambda labels: bool(set(labels) & chosen_classes_set))].index.tolist()
        
#         # Create a DataFrame with these indices
#         cls_indices_df = self.dataframe.iloc[cls_indices]
        
#         if len(cls_indices) < self.k_shot + self.q_query:
#             raise ValueError("Not enough samples to create an episode with the chosen classes.")
        
#         support_indices = []
#         for i in chosen_classes:    
#             temp_df = cls_indices_df[cls_indices_df['label'].apply(lambda x: i in x)]
#             selected_rows = temp_df.sample(n=self.k_shot)
#             selected_indices = selected_rows.index.tolist() 
#             support_indices += selected_indices  
        
#         support_indices = list(set(support_indices))

#         query_indices = np.random.choice(cls_indices, self.q_query, replace=False)
        
#         for idx in support_indices:
#             if idx < 0 or idx >= len(self.dataframe):
#                 raise IndexError(f"Support index {idx} is out-of-bounds")
#             img, label = self.__getitem__(idx)
#             support_x.append(img)
#             support_y.append(label)

#         for idx in query_indices:
#             if idx < 0 or idx >= len(self.dataframe):
#                 raise IndexError(f"Query index {idx} is out-of-bounds")
#             img, label = self.__getitem__(idx)
#             query_x.append(img)
#             query_y.append(label)

#         return torch.stack(support_x), torch.stack(support_y), torch.stack(query_x), torch.stack(query_y)

# # Epdisode를 만들어주는 DataLoader 추가
# def episodic_dataloader(dataset, num_episodes):
#     for _ in range(num_episodes):
#         yield dataset.create_episode()


class EpisodicNMCDataset:
    def __init__(self, root_dir, n_way, k_shot, q_query, split_ratio=0.75, minor_cls=[4, 5], transform=None):
        self.image_dir = root_dir
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.split_ratio = split_ratio
        self.minor_cls = set(minor_cls)
        
        df_path = os.path.join(root_dir, 'nmc_combined.csv')

        combined_df = pd.read_csv(df_path).dropna()
        
        # Process labels to list format
        def process_label(x):
            if isinstance(x, str):
                return x.split(',')
            else:
                raise ValueError(f"Unexpected label value: {x}")
        
        combined_df['label'] = combined_df['label'].apply(process_label)
        
        # Filter labels to only include valid values between 0 and 10
        def filter_labels(labels):
            return [int(label) for label in labels if label and 0 <= int(label) <= 10]
        
        combined_df['label'] = combined_df['label'].apply(filter_labels)

        # Separate minor class samples for test set
        minor_class_df = combined_df[combined_df['label'].apply(lambda labels: bool(self.minor_cls & set(labels)))]
        
        # Separate non-minor class samples
        non_minor_class_df = combined_df[~combined_df['label'].apply(lambda labels: bool(self.minor_cls & set(labels)))]
        
        # Split non-minor class data into training and test set
        train_size = int(len(non_minor_class_df) * split_ratio)
        
        train_df = non_minor_class_df.sample(n=train_size, random_state=42)  # Training set without minor classes
        remaining_non_minor_df = non_minor_class_df.drop(train_df.index)  # Remaining non-minor classes for test

        # Combine remaining non-minor samples with minor class samples for test set
        test_df = pd.concat([remaining_non_minor_df, minor_class_df], ignore_index=True)  # Test set with both minor and non-minor classes

        # Initialize the MultiLabelBinarizer and fit_transform the label column for both sets
        self.mlb = MultiLabelBinarizer(classes=[int(i) for i in range(11)])
        
        # Prepare training data
        self.train_df = train_df.reset_index(drop=True)
        self.train_labels = self.mlb.fit_transform(self.train_df['label'])
        
        self.train_unique_labels = sorted(list(set(sum(self.train_df['label'].tolist(), []))))
        
        
        # Prepare test data
        self.test_df = test_df.reset_index(drop=True)
        self.test_labels = self.mlb.transform(self.test_df['label'])  # Fit only on training data
        
        self.test_unique_labels = sorted(list(set(sum(self.test_df['label'].tolist(), []))))

        # Store number of classes
        self.n_classes = len(self.mlb.classes_)

        print("Training set size:", len(self.train_df))
        print("Test set size:", len(self.test_df))
        

    def get_train_test_split(self):
        """
        Returns the training and test datasets.
        """
        # Return training and test data separately
        return EpisodicDataSubset(self.train_df, self.train_labels, self.image_dir, self.transform, self.n_way, self.k_shot, self.q_query, self.n_classes, self.train_unique_labels), \
               EpisodicDataSubset(self.test_df, self.test_labels, self.image_dir, self.transform, self.n_way, self.k_shot, self.q_query, self.n_classes, self.test_unique_labels)


class EpisodicDataSubset(Dataset):
    def __init__(self, dataframe, labels, image_dir, transform=None, n_way=5, k_shot=5, q_query=15, n_classes=None, unique_label=None):
        self.dataframe = dataframe
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_classes = n_classes  # Set n_classes attribute
        self.unique_label = unique_label

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.dataframe):
            raise IndexError(f"Index {idx} is out-of-bounds")
        
        img_name = self.dataframe.iloc[idx]['image']
        img_path = os.path.join(self.image_dir, 'combined_images', img_name)
        image = read_image(img_path)
        
        label_vector = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_vector, dtype=torch.float32)

    def create_episode(self):
        support_x = []
        support_y = []
        query_x = []
        query_y = []

        # Randomly choose classes for the episode
        chosen_classes = np.random.choice(self.unique_label, self.n_way, replace=False)
        chosen_classes_set = set(chosen_classes)
        
        print(f'chosen_Cls: {chosen_classes_set}')

        # Find all indices where the label contains at least one of the chosen classes
        cls_indices = self.dataframe[self.dataframe['label'].apply(lambda labels: bool(set(labels) & chosen_classes_set))].index.tolist()
        # Create a DataFrame with these indices
        cls_indices_df = self.dataframe.iloc[cls_indices]
        # Check if there are enough samples to create an episode
        if len(cls_indices) < self.k_shot + self.q_query:
            raise ValueError("Not enough samples to create an episode with the chosen classes.")

        support_indices = []
        for i in chosen_classes:
            temp_df = cls_indices_df[cls_indices_df['label'].apply(lambda x: i in x)]
            
            # Check if there are enough samples for the support set
            if len(temp_df) < self.k_shot:
                print(f"Not enough samples for class {i} to create a support set. Skipping this class.")
                continue
            
            # Sample k_shot instances for the support set
            selected_rows = temp_df.sample(n=self.k_shot)
            selected_indices = selected_rows.index.tolist()
            support_indices += selected_indices

        # If not enough support samples were found, raise an error
        if len(support_indices) < self.k_shot * self.n_way:
            raise ValueError("Not enough support samples to create an episode. Try increasing the dataset size or adjusting k_shot.")

        support_indices = list(set(support_indices))

        # Ensure enough samples are available for the query set after selecting support samples
        available_indices = list(set(cls_indices) - set(support_indices))
        if len(available_indices) < self.q_query:
            raise ValueError("Not enough samples available for the query set. Adjust the dataset size or k_shot and q_query values.")
        
        # Sample q_query instances for the query set from the remaining indices
        query_indices = np.random.choice(available_indices, self.q_query, replace=False)
        
        # Prepare the support set
        for idx in support_indices:
            img, label = self.__getitem__(idx)
            support_x.append(img)
            support_y.append(label)

        # Prepare the query set
        for idx in query_indices:
            img, label = self.__getitem__(idx)
            query_x.append(img)
            query_y.append(label)

        return torch.stack(support_x), torch.stack(support_y), torch.stack(query_x), torch.stack(query_y)


# Epdisode를 만들어주는 DataLoader 추가
def episodic_dataloader(dataset, num_episodes):
    for _ in range(num_episodes):
        yield dataset.create_episode()