from torch import nn
import torch
class MultiHeadEmbedding(nn.Module):
    def __init__(self, num_features, num_classes, embedding_dim):
        super().__init__()
        # 클래스 수만큼의 임베딩 헤드 생성
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, embedding_dim)
            ) for _ in range(num_classes)
        ])
    
    def forward(self, x):
        # 각 헤드별로 임베딩 생성
        embeddings = [head(x) for head in self.heads]
        return torch.stack(embeddings, dim=1)  # (batch_size, num_classes, embedding_dim)
