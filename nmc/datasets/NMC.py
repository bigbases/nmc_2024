import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import os 
import numpy as np 
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.io import read_image
from itertools import chain
import sys
import itertools


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

        # Separate minor and major classes
        minor_class_df = combined_df[combined_df['label'].apply(lambda labels: bool(self.minor_cls & set(labels)))]
        non_minor_class_df = combined_df[~combined_df['label'].apply(lambda labels: bool(self.minor_cls & set(labels)))]

        # Train-test split
        train_size = int(len(combined_df) * split_ratio)
        train_df = non_minor_class_df.sample(n=train_size, random_state=12) 
        remaining_non_minor_df = non_minor_class_df.drop(train_df.index)

        test_df = pd.concat([remaining_non_minor_df, minor_class_df], ignore_index=True) 

        # Initialize MultiLabelBinarizer
        self.mlb = MultiLabelBinarizer(classes=[int(i) for i in range(11)])

        # Train and Test sets with labels
        self.train_df = train_df.reset_index(drop=True)
        self.train_labels = self.mlb.fit_transform(self.train_df['label'])
        self.train_unique_labels = sorted(list(set(sum(self.train_df['label'].tolist(), []))))
        
        self.test_df = test_df.reset_index(drop=True)
        self.test_labels = self.mlb.transform(self.test_df['label'])
        self.test_unique_labels = sorted(list(set(sum(self.test_df['label'].tolist(), []))))

        # Store number of classes
        self.n_classes = len(self.mlb.classes_)

        # Pre-generate all possible episodes for train and test sets
        self.train_episodes = self._pre_generate_train_episodes()
        self.test_episodes = self._pre_generate_test_episodes()
        self.train_episode_counter = 0  # To keep track of current train episode
        self.test_episode_counter = 0   # To keep track of current test episode

        print("Training set size:", len(self.train_df))
        print("Test set size:", len(self.test_df))
        print(f'Train unique label: {self.train_unique_labels}')
        print(f'Test unique label: {self.test_unique_labels}')
        print(f'Training episode size: {len(self.train_episodes)}')
        print(f'Test episode size: {len(self.test_episodes)}')
        print(self.train_df['label'].explode().value_counts())  # 각 클래스의 샘플 수를 출력
        print(self.test_df['label'].explode().value_counts())  # 각 클래스의 샘플 수를 출력


    def _pre_generate_train_episodes(self):
        episodes = []
        unique_labels = self.train_unique_labels
        combinations = list(itertools.combinations(unique_labels, self.n_way))

        for combination in combinations:
            support_indices = []
            query_indices = []
            already_selected_indices = set()  # Track selected samples for support set

            # Support set creation
            for cls in combination:
                # Include all samples for the class (allow duplicates in support)
                cls_indices = self.train_df[self.train_df['label'].apply(lambda x: cls in x)].index.tolist()
                
                if len(cls_indices) < self.k_shot:
                    print(f"Not enough samples for class {cls} in combination {combination} for support set. Needed: {self.k_shot}, Available: {len(cls_indices)}")
                    print("Selected Train Support Indices DataFrame Rows:")
                    print(self.train_df.loc[support_indices])  # 선택된 인덱스의 행을 출력
                    break  # Skip if not enough samples for k-shot

                # 중복 허용 샘플 선택
                selected_support = np.random.choice(cls_indices, self.k_shot, replace=False)
                support_indices.extend(selected_support)
                already_selected_indices.update(selected_support)  # Needed to avoid duplicates in query set

            # Ensure enough support samples are selected
            if len(support_indices) < self.k_shot * self.n_way:
                print(f"Not enough support samples for combination {combination}. Needed: {self.k_shot * self.n_way}, Available: {len(support_indices)}")
                continue

            # Query set creation: must not overlap with support set or within query set
            remaining_indices = list(set(self.train_df.index) - already_selected_indices)
            remaining_df = self.train_df.loc[remaining_indices]

            already_selected_for_query = set()  # Initialize globally for all classes

            for cls in combination:
                # Query set에서 중복을 허용하지 않음
                cls_indices = remaining_df[remaining_df['label'].apply(lambda x: cls in x) & ~remaining_df.index.isin(already_selected_for_query)].index.tolist()
                
                if len(cls_indices) < self.q_query // self.n_way:
                    print(f"Not enough samples for class {cls} in combination {combination} for query set. Needed: {self.q_query // self.n_way}, Available: {len(cls_indices)}")
                    break  # Skip if not enough samples for q-query

                selected_query = np.random.choice(cls_indices, self.q_query // self.n_way, replace=False)
                query_indices.extend(selected_query)
                already_selected_for_query.update(selected_query)  # Update globally

            # Ensure enough query samples are selected
            if len(query_indices) < self.q_query:
                print(f"Not enough query samples for combination {combination}. Needed: {self.q_query}, Available: {len(query_indices)}")
                continue

            # Save the episode
            episodes.append((support_indices, query_indices))

        print(f"Generated {len(episodes)} train episodes.")  # Log the number of generated episodes
        return episodes


    def _pre_generate_test_episodes(self):
        episodes = []
        minor_labels = list(self.minor_cls)  # [4, 5]
        major_labels = [label for label in self.test_unique_labels if label not in self.minor_cls]

        # Generate combinations ensuring minor classes are included
        for major_comb in itertools.combinations(major_labels, self.n_way - len(minor_labels)):
            combination = minor_labels + list(major_comb)  # Ensure 4 and 5 are included in every combination
            support_indices = []
            query_indices = []
            already_selected_indices = set()

            # Support set creation
            for cls in combination:
                # Exclude already selected samples to avoid duplicates
                cls_indices = self.test_df[self.test_df['label'].apply(lambda x: cls in x) & ~self.test_df.index.isin(already_selected_indices)].index.tolist()
                if len(cls_indices) < self.k_shot:
                    print("Selected Test Support Indices DataFrame Rows:")
                    print(self.test_df.loc[support_indices])  # 선택된 인덱스의 행을 출력
                    continue  # Skip if not enough samples for k-shot

                selected_support = np.random.choice(cls_indices, self.k_shot, replace=False)
                support_indices.extend(selected_support)
                already_selected_indices.update(selected_support)

            # Ensure enough support samples are selected
            if len(support_indices) < self.k_shot * self.n_way:
                continue

            # Query set creation: must not overlap with support set or within query set
            remaining_indices = list(set(self.test_df.index) - already_selected_indices)
            remaining_df = self.test_df.loc[remaining_indices]

            # 전역적으로 중복 방지를 위한 집합
            already_selected_for_query = set()  # Initialize globally for all classes

            for cls in combination:
                # 모든 클래스에 대해 선택된 샘플을 전역적으로 중복 방지하도록 수정
                cls_indices = remaining_df[remaining_df['label'].apply(lambda x: cls in x) & ~remaining_df.index.isin(already_selected_for_query)].index.tolist()
                if len(cls_indices) < self.q_query // self.n_way:
                    continue  # Skip if not enough samples for q-query

                selected_query = np.random.choice(cls_indices, self.q_query // self.n_way, replace=False)
                query_indices.extend(selected_query)
                already_selected_for_query.update(selected_query)  # Update globally

            # Ensure enough query samples are selected
            if len(query_indices) < self.q_query:
                
                continue

            # Save the episode
            episodes.append((support_indices, query_indices))

        return episodes

    def create_episode(self, is_train=True):
        """
        Get a pre-generated episode for train or test set.
        """
        if is_train:
            if self.train_episode_counter >= len(self.train_episodes):
                raise IndexError("All training episodes have been used. Reset the episode counter or extend episodes.")
            
            support_indices, query_indices = self.train_episodes[self.train_episode_counter]
            self.train_episode_counter += 1
        else:
            if self.test_episode_counter >= len(self.test_episodes):
                raise IndexError("All testing episodes have been used. Reset the episode counter or extend episodes.")
            
            support_indices, query_indices = self.test_episodes[self.test_episode_counter]
            self.test_episode_counter += 1
        
        # Fetch images and labels for support and query sets
        support_x, support_y = self._fetch_data(support_indices, is_train)
        query_x, query_y = self._fetch_data(query_indices, is_train)
        
        return support_x, support_y, query_x, query_y

    def _fetch_data(self, indices, is_train=True):
        """
        Fetch data for given indices.
        """
        dataframe = self.train_df if is_train else self.test_df
        labels = self.train_labels if is_train else self.test_labels

        images, labels_out = [], []
        for idx in indices:
            img_name = dataframe.iloc[idx]['image']
            img_path = os.path.join(self.image_dir, 'combined_images', img_name)
            image = read_image(img_path)
            label_vector = labels[idx]

            if self.transform:
                image = self.transform(image)

            images.append(image)
            labels_out.append(torch.tensor(label_vector, dtype=torch.float32))

        return torch.stack(images), torch.stack(labels_out)
