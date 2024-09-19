from torch.utils.data import Dataset, DataLoader
from torchvision import io
from PIL import Image
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
import os
import torch
from torchvision.io import read_image

class ODIRDataset(Dataset):
    def __init__(self, image_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, transform=None):
        print(image_dir)
        
        # Assuming combined CSV file path
        df_path = image_dir.replace('combined_images', 'cropped_combined.csv')
        
        # Load the dataframe and preprocess labels
        self.dataframe = pd.read_csv(df_path)
        self.dataframe = self.dataframe.dropna()

        # Convert label columns to a list of indices for multilabel binarization
        def process_labels(row):
            labels = []
            for i, col in enumerate(self.dataframe.columns[6:-1]):  # Assuming label columns start from 6 to end-1
                if row[col] == 1:
                    labels.append(i)
            return labels

        self.dataframe['label'] = self.dataframe.apply(process_labels, axis=1)
        
        # Initialize the MultiLabelBinarizer and fit_transform the label column
        self.CLASSES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        self.mlb = MultiLabelBinarizer(classes=[i for i in range(len(self.CLASSES))])
        labels = self.mlb.fit_transform(self.dataframe['label'])

        # Perform iterative stratified split for train (70%) and remaining (30%)
        X = self.dataframe['Fundus'].values.reshape(-1, 1)  # Image paths as features
        y = labels  # Multi-hot encoded labels

        X_train, y_train, X_remaining, y_remaining = iterative_train_test_split(X, y, test_size=(1 - train_ratio))

        # Further split remaining data into val (50% of remaining) and test (50% of remaining)
        X_val, y_val, X_test, y_test = iterative_train_test_split(X_remaining, y_remaining, test_size=(test_ratio / (valid_ratio + test_ratio)))

        # Convert the split data back to DataFrames
        train_df = pd.DataFrame({'image': X_train.flatten(), 'label': self.mlb.inverse_transform(y_train)})
        val_df = pd.DataFrame({'image': X_val.flatten(), 'label': self.mlb.inverse_transform(y_val)})
        test_df = pd.DataFrame({'image': X_test.flatten(), 'label': self.mlb.inverse_transform(y_test)})
        
        print(train_df['label'].value_counts())
        print(f'Train size: {len(train_df)}')
        print(val_df['label'].value_counts())
        print(f'Validation size: {len(val_df)}')
        print(test_df['label'].value_counts())
        print(f'Test size: {len(test_df)}')
        
        # Store the split dataframes and labels
        self.train_data = (train_df, y_train)
        self.val_data = (val_df, y_val)
        self.test_data = (test_df, y_test)

        self.n_classes = len(self.CLASSES)
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Fundus']
        img_path = os.path.join(self.image_dir, img_name)
        image = read_image(img_path)
        
        label_vector = self.mlb.transform([self.dataframe.iloc[idx]['label']])[0]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_vector, dtype=torch.float32)

    def get_splits(self):
        """ Return train, val, test datasets as ODIRDataset instances """
        train_df, train_labels = self.train_data
        val_df, val_labels = self.val_data
        test_df, test_labels = self.test_data
        
        train_dataset = ODIRDatasetSplit(train_df, train_labels, self.n_classes, self.image_dir, self.transform)
        val_dataset = ODIRDatasetSplit(val_df, val_labels, self.n_classes, self.image_dir, self.transform)
        test_dataset = ODIRDatasetSplit(test_df, test_labels, self.n_classes, self.image_dir, self.transform)
        
        return train_dataset, val_dataset, test_dataset


class ODIRDatasetSplit(Dataset):
    """Helper class for creating dataset instances for train, val, and test splits"""
    def __init__(self, dataframe, labels, n_classes, image_dir, transform=None):
        self.dataframe = dataframe
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform
        self.n_classes = n_classes
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image']
        img_path = os.path.join(self.image_dir, img_name)
        image = read_image(img_path)
        
        label_vector = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_vector, dtype=torch.float32)

