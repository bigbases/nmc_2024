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

        # 테스트용 minor cls DF
        minor_class_df = combined_df[combined_df['label'].apply(lambda labels: bool(self.minor_cls & set(labels)))]
        
        # 전체 major cls DF
        non_minor_class_df = combined_df[~combined_df['label'].apply(lambda labels: bool(self.minor_cls & set(labels)))]
        
        # 전체 DF에서 비율에 따라 분리
        train_size = int(len(combined_df) * split_ratio)
        
        # major cls DF에서 훈련 크기만큼 샘플링
        train_df = non_minor_class_df.sample(n=train_size, random_state=42) 
        # 나머지 major cls 샘플들 (테스트에 할당 예정)
        remaining_non_minor_df = non_minor_class_df.drop(train_df.index)

        # minor cls DF + 나머지 major cls 샘플
        test_df = pd.concat([remaining_non_minor_df, minor_class_df], ignore_index=True) 

        # Initialize the MultiLabelBinarizer and fit_transform the label column for both sets
        self.mlb = MultiLabelBinarizer(classes=[int(i) for i in range(11)])
        

        self.train_df = train_df.reset_index(drop=True)
        self.train_labels = self.mlb.fit_transform(self.train_df['label'])
        # Trainset의 고유 라벨들
        self.train_unique_labels = sorted(list(set(sum(self.train_df['label'].tolist(), []))))
        
        self.test_df = test_df.reset_index(drop=True)
        self.test_labels = self.mlb.transform(self.test_df['label'])  # Fit only on training data
        # Testset의 고유 라벨들
        self.test_unique_labels = sorted(list(set(sum(self.test_df['label'].tolist(), []))))

        # Store number of classes
        self.n_classes = len(self.mlb.classes_)

        print("Training set size:", len(self.train_df))
        print("Test set size:", len(self.test_df))
        print(f'Train unique label: {self.train_unique_labels}')
        print(f'Test unique label: {self.test_unique_labels}')
        
    def get_train_test_split(self):
        """
        Returns the training and test datasets.
        """
        # Return training and test data separately
        return EpisodicDataSubset(self.train_df, self.train_labels, self.image_dir, self.minor_cls, self.transform, self.n_way, self.k_shot, self.q_query, self.n_classes, self.train_unique_labels, True), \
               EpisodicDataSubset(self.test_df, self.test_labels, self.image_dir, self.minor_cls, self.transform, self.n_way, self.k_shot, self.q_query, self.n_classes, self.test_unique_labels, False)


class EpisodicDataSubset(Dataset):
    def __init__(self, dataframe, labels, image_dir, minor_cls=[4,5],transform=None, n_way=5, k_shot=5, q_query=15, n_classes=None, unique_label=None, is_train=True):
        self.dataframe = dataframe
        self.labels = labels
        self.image_dir = image_dir
        self.minor_cls = minor_cls
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_classes = n_classes  # Set n_classes attribute
        self.unique_label = unique_label
        self.is_train = is_train

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

        if not self.is_train:  # Test case
            chosen_classes = list(self.minor_cls)  
            remaining_classes_needed = self.n_way - len(self.minor_cls)
            remaining_classes = [label for label in self.unique_label if label not in self.minor_cls]
            
            if remaining_classes_needed > 0:
                additional_classes = list(np.random.choice(remaining_classes, remaining_classes_needed, replace=False))
                chosen_classes.extend(additional_classes)
                
            # minor cls 포함 n에 해당하는 샘플들
            cls_indices_df = self.dataframe[self.dataframe['label'].apply(lambda labels: bool(set(labels) & set(chosen_classes)))]
            
        else:  # Train case
            chosen_classes = np.random.choice(self.unique_label, self.n_way, replace=False).tolist()
            chosen_classes_set = set(chosen_classes)
            # 매 episode마다 랜덤으로 n개의 클래스 샘플링
            cls_indices_df = self.dataframe[self.dataframe['label'].apply(lambda labels: bool(set(labels) & chosen_classes_set))]

        if len(cls_indices_df) < self.k_shot + self.q_query:
            raise ValueError("Not enough samples to create an episode with the chosen classes.")

        support_indices = []
        already_selected_indices = set()

        # support set 생성 과정
        for i in chosen_classes:
            # 이미 사용한 샘플 제외
            temp_df = cls_indices_df[cls_indices_df['label'].apply(lambda x: i in x) & ~cls_indices_df.index.isin(already_selected_indices)]
            
            if len(temp_df) < self.k_shot:
                print(f"Not enough samples for class {i} to create a support set. Skipping this class.")
                continue
            
            # n에 대해 k개만큼 샘플링
            selected_rows = temp_df.sample(n=self.k_shot)
            selected_indices = selected_rows.index.tolist()
            support_indices += selected_indices
            
            # 사용한 샘플 추가 (중복 방지)
            already_selected_indices.update(selected_indices)

        if len(support_indices) < self.k_shot * self.n_way:
            raise ValueError("Not enough support samples to create an episode. Try increasing the dataset size or adjusting k_shot.")

        # support set
        support_indices = list(set(support_indices))
        # Support에서 사용한 샘플 제외 (for query set)
        available_indices = list(set(cls_indices_df.index) - set(support_indices))
        
        if self.is_train: #Train Case
            if len(available_indices) < self.q_query:
                raise ValueError("Not enough samples available for the query set. Adjust the dataset size or k_shot and q_query values.")
            # Trainset에서 Support에 사용하지 않은 샘플들로 샘플링 (minor class 제외하고 클래스에 상관없이, support set과 같은 클래스 종류)
            query_indices = np.random.choice(available_indices, self.q_query, replace=False)
        else:
            # Testset에서 Support에 사용하지 않은 샘플들로 샘플링 (minor class는 무조건 포함, support set과 같은 클래스 종류)
            remaining_indices_df = self.dataframe.loc[available_indices]
            query_indices = []

            # 모든 클래스는 균등하게 배분 ([3,4,5]인 3-way, 10-query인 경우 3개씩 할당 후 나머지 한 개는 랜덤 할당)
            num_samples_per_class = self.q_query // self.n_way  
            num_remaining_samples = self.q_query % self.n_way

            for cls in chosen_classes:
                class_samples = remaining_indices_df[remaining_indices_df['label'].apply(lambda labels: cls in labels)].index.tolist()

                if len(class_samples) < num_samples_per_class:
                    raise ValueError(f"Not enough samples for class {cls} to create a query set. Adjust the dataset size or q_query values.")

                selected_samples = np.random.choice(class_samples, num_samples_per_class, replace=False)
                query_indices.extend(selected_samples)

            remaining_samples_pool = list(set(available_indices) - set(query_indices))

            if len(remaining_samples_pool) < num_remaining_samples:
                raise ValueError("Not enough samples available to fill the remaining query set. Adjust the dataset size or q_query values.")

            remaining_query_indices = np.random.choice(remaining_samples_pool, num_remaining_samples, replace=False)
            query_indices.extend(remaining_query_indices)
            query_indices = np.array(query_indices)

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