from torch.utils.data import DataLoader
from torch import nn, optim
import pandas as pd
from nmc.models.fgmaxvit import FGMaxxVit
from nmc.datasets.APTOS import *


def train(model, dataset, optimizer, criterion, num_episodes):
    model.train()
    # Episode 마다 학습
    for episode in episodic_dataloader(dataset, num_episodes):
        support_x, support_y, query_x, query_y = episode

        optimizer.zero_grad()
        support_pred = model(support_x)
        loss = criterion(support_pred, support_y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval() 
            query_pred = model(query_x)
            query_loss = criterion(query_pred, query_y)
            print(f'Query Loss: {query_loss.item()}')

        model.train() 


if __name__ == '__main__':
    model = FGMaxxVit()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataframe = pd.read_csv('your_dataframe.csv')
    image_dir = 'your_image_directory'
    # transformations = None

    episodic_dataset = EpisodicAPTOSDataset(dataframe, image_dir, n_way=5, k_shot=5, q_query=15, transform=transformations)
    train(model, episodic_dataset, optimizer, criterion, num_episodes=100)