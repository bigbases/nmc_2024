import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import io  # Assuming this is for image reading
from sklearn.model_selection import train_test_split

class APTOSDataset(Dataset):
    def __init__(self, image_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, transform=None, target_label=None):
        print(image_dir)
        
        # Defining the classes for classification
        self.CLASSES = ['Normal', 'Mild', 'Moderate Disease Level', 'Server', 'Proliferative']
        self.n_classes = len(self.CLASSES)
        self.target_label = target_label

        # Assuming combined CSV file path
        df_path = image_dir.replace('combined_images', 'combined.csv')
        
        # Read the combined CSV file
        self.dataframe = pd.read_csv(df_path)
        self.dataframe = self.dataframe.dropna()

        # Convert 'diagnosis' column to integer (if it's not already)
        self.dataframe['diagnosis'] = self.dataframe['diagnosis'].astype(int)

        # Perform stratified split for train (70%), validation (15%), and test (15%)
        X = self.dataframe['id_code'].values  # Image paths as features
        y = self.dataframe['diagnosis'].values  # Integer labels
        
        # print(self.dataframe['diagnosis'].head(50))
        # if self.target_label is not None:
        #     self.dataframe['diagnosis'] = self.dataframe['diagnosis'].apply(lambda x: 1 if self.target_label == x else 0)
        #     y = self.dataframe['diagnosis'].values
        # print(self.dataframe['diagnosis'].head(50))
        
        # Split the data into training and remaining (validation + test)
        X_train, X_remaining, y_train, y_remaining = train_test_split(
            X, y, test_size=(1 - train_ratio), stratify=y, random_state=42
        )

        # Further split remaining data into validation and test
        X_val, X_test, y_val, y_test = train_test_split(
            X_remaining, y_remaining, test_size=(test_ratio / (valid_ratio + test_ratio)), stratify=y_remaining, random_state=42
        )

        # Convert the split data back to DataFrames
        train_df = pd.DataFrame({'id_code': X_train, 'diagnosis': y_train})
        val_df = pd.DataFrame({'id_code': X_val, 'diagnosis': y_val})
        test_df = pd.DataFrame({'id_code': X_test, 'diagnosis': y_test})
        
        print(train_df['diagnosis'].value_counts())
        print(f'Train size: {len(train_df)}')
        print(val_df['diagnosis'].value_counts())
        print(f'Validation size: {len(val_df)}')
        print(test_df['diagnosis'].value_counts())
        print(f'Test size: {len(test_df)}')
        
        # Store the split dataframes and labels
        self.train_data = (train_df, y_train)
        self.val_data = (val_df, y_val)
        self.test_data = (test_df, y_test)

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the image ID and label
        img_name = self.dataframe.iloc[idx]['id_code']
        img_name += '.png'
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read the image
        image = io.read_image(img_path)
        
        # Get the label (integer label for single-label classification)
        label = self.dataframe.iloc[idx]['diagnosis']

        # Create a one-hot encoded vector for the label
        # one_hot = np.zeros(self.n_classes, dtype=np.float32)
        # one_hot[label] = 1.0

        if self.transform:
            image = self.transform(image)

        # Return the image and the one-hot encoded label
        return image, torch.tensor(one_hot, dtype=torch.float32)

    def get_splits(self):
        """Return train, val, test datasets as separate APTOSDataset instances."""
        train_df, train_labels = self.train_data
        val_df, val_labels = self.val_data
        test_df, test_labels = self.test_data
        
        train_dataset = APTOSDatasetSplit(train_df, train_labels, self.n_classes, self.image_dir, self.transform)
        val_dataset = APTOSDatasetSplit(val_df, val_labels, self.n_classes, self.image_dir, self.transform)
        test_dataset = APTOSDatasetSplit(test_df, test_labels, self.n_classes, self.image_dir, self.transform)
        
        return train_dataset, val_dataset, test_dataset

class APTOSDatasetSplit(Dataset):
    """A helper class for managing split datasets."""
    def __init__(self, dataframe, labels, n_classes, image_dir, transform=None):
        self.dataframe = dataframe
        self.labels = labels
        self.n_classes = n_classes
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['id_code']
        img_name += '.png'
        img_path = os.path.join(self.image_dir, img_name)
        
        # Read the image
        image = io.read_image(img_path)
        
        # Get the label (direct integer label)
        label = self.labels[idx]

        # Create a one-hot encoded vector for the label
        one_hot = np.zeros(self.n_classes, dtype=np.float32)
        one_hot[label] = 1.0

        if self.transform:
            image = self.transform(image)

        # return image, torch.tensor(label, dtype=torch.float32)
        return image, torch.tensor(one_hot, dtype=torch.float32)

class EpisodicAPTOSDataset(Dataset):
    def __init__(self, image_dir, n_way, k_shot, q_query, transform=None):
        super().__init__()
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
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.CLASSES = ['Normal', 'Mild', 'Moderate Disease Level', 'Severe', 'Proliferative']
        self.n_classes = len(self.CLASSES)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['id_code'] + '.png'
        img_path = os.path.join(self.image_dir, img_name)
        image = read_image(img_path)

        one_hot = np.zeros(self.n_classes)
        one_hot[self.dataframe.iloc[idx]['diagnosis']] = 1

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(one_hot)

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

        return torch.stack(support_x), torch.stack(support_y), torch.stack(query_x), torch.stack(query_y)
    


# Epdisode를 만들어주는 DataLoader 추가
def episodic_dataloader(dataset, num_episodes):
    for _ in range(num_episodes):
        yield dataset.create_episode()
