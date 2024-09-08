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
    def __init__(self, root_dir, n_way, k_shot, q_query, episodes, split_ratio=0.75, minor_cls=[4, 5], transform=None):
        self.image_dir = root_dir
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.split_ratio = split_ratio
        self.minor_cls = set(minor_cls)
        self.num_episodes = episodes
        
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
        used_samples = set()

        # 모든 조합에 대해 episode 생성
        for combination in combinations:
            # allow_reuse는 에피소드 간 중복 여부 (1번 에피소드에 등장한 샘플이 2번 에피소드에 등장 할 수 있음)
            # 모든 조합에 대해선 에피소드 간 중복 허용 x (다양성을 위해)
            support_indices, query_indices = self._generate_episode(combination, used_samples, self.train_df, allow_reuse=False)
            if support_indices and query_indices:
                episodes.append((support_indices, query_indices))

        # 미사용 샘플 사용하기 위해 랜덤 조합으로 episode 생성
        while len(used_samples) < len(self.train_df):
            random_classes = np.random.choice(unique_labels, self.n_way, replace=False)
            # 에피소드 간 중복 허용
            support_indices, query_indices = self._generate_episode(random_classes, used_samples, self.train_df, allow_reuse=True)
            if support_indices and query_indices:
                episodes.append((support_indices, query_indices))
        
        final_num_episodes = self.num_episodes - len(episodes)
        
        # 모든 샘플을 적어도 한 번씩 사용한 후 지정한 에피소드 개수만큼 랜덤으로 에피소드 생성 
        for _ in range(final_num_episodes):
            random_classes = np.random.choice(unique_labels, self.n_way, replace=False)
            # 에피소드 간 중복 허용
            support_indices, query_indices = self._generate_episode(random_classes, used_samples, self.train_df, allow_reuse=True)
            if support_indices and query_indices:
                episodes.append((support_indices, query_indices))

        print(f"Generated {len(episodes)} train episodes.")
        return episodes

    def _pre_generate_test_episodes(self):
        episodes = []
        minor_labels = list(self.minor_cls)  # 소수 클래스 무조건 포함
        major_labels = [label for label in self.test_unique_labels if label not in self.minor_cls]
        used_samples = set()  # 사용된 샘플 트래킹

        # 소수 클래스를 무조건 포함하는 모든 조합에 대해 episode 생성
        for major_comb in itertools.combinations(major_labels, self.n_way - len(minor_labels)):
            combination = minor_labels + list(major_comb)
            # 에피소드 간 중복 허용
            support_indices, query_indices = self._generate_episode(combination, used_samples, self.test_df, allow_reuse=True)
            if support_indices and query_indices:
                episodes.append((support_indices, query_indices))

        return episodes

    def _generate_episode(self, classes, used_samples, df, allow_reuse=False):
        support_indices, query_indices = [], []
        already_selected_indices = set()

        # Support set 생성
        for cls in classes:
            # 이미 쓰인 샘플 제외
            cls_indices = df[df['label'].apply(lambda x: cls in x) & 
                            ~df.index.isin(already_selected_indices) & 
                            ~df.index.isin(used_samples)].index.tolist()

            # 이미 쓰인 샘플 제외하고 shot 개수를 충족하지 못하면 중복 샘플링 허용 
            if len(cls_indices) < self.k_shot and allow_reuse:
                cls_indices = df[df['label'].apply(lambda x: cls in x) & 
                                ~df.index.isin(already_selected_indices)].index.tolist()

            # support 내에서 중복 허용하지 않고 샘플링
            if len(cls_indices) >= self.k_shot:
                selected_support = np.random.choice(cls_indices, self.k_shot, replace=False)
                support_indices.extend(selected_support)
                already_selected_indices.update(selected_support)
                used_samples.update(selected_support)

        # 개수 만족하는 지 확인
        if len(support_indices) < self.k_shot * self.n_way:
            return None, None

        # Query set 생성
        # support에서 쓰인 것 제외
        remaining_indices = list(set(df.index) - already_selected_indices)
        remaining_df = df.loc[remaining_indices]
        already_selected_for_query = set()

        # 이미 쓰인 것 제외하고 추출
        for cls in classes:
            cls_indices = remaining_df[remaining_df['label'].apply(lambda x: cls in x) & 
                                    ~remaining_df.index.isin(already_selected_for_query)].index.tolist()

            if len(cls_indices) < self.q_query // self.n_way:
                continue 
            
            # query 내에서 중복 허용하지 않고 샘플링
            selected_query = np.random.choice(cls_indices, self.q_query // self.n_way, replace=False)
            query_indices.extend(selected_query)
            already_selected_for_query.update(selected_query)
            used_samples.update(selected_query)

        # 개수 만족하는 지 확인
        if len(query_indices) < self.q_query:
            return None, None

        return support_indices, query_indices

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
