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

class MultiHeadEmbeddingBCE(nn.Module):
    def __init__(self, num_features, num_classes, embedding_dim):
        super().__init__()
        # 공유되는 기본 변환 layer (optional)
        self.shared = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, num_features),
            nn.ReLU()
        )
        
        # 독립적인 head와 classifier
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Linear(512, embedding_dim)
            ) for _ in range(num_classes)
        ])
        
        self.classifiers = nn.ModuleList([
            nn.Linear(embedding_dim, 1) for _ in range(num_classes)
        ])
        
        # 레이블 상관관계 (gradient 계산에는 영향을 주지 않음)
        self.register_buffer('label_correlations', torch.eye(num_classes))
    
    def update_label_correlations(self, labels):
        """레이블 상관관계 업데이트"""
        with torch.no_grad():
            co_occurrence = torch.matmul(labels.t(), labels).float()
            total_occurrences = co_occurrence.diagonal().unsqueeze(1)
            normalized = co_occurrence / (total_occurrences + 1e-8)
            self.label_correlations = 0.9 * self.label_correlations + 0.1 * normalized
    
    def forward(self, x, labels=None):
        # 공유 특징 추출
        shared_features = self.shared(x)
        
        # 각 head의 임베딩 계산
        embeddings = []
        logits = []
        
        for i in range(len(self.heads)):
            class_embedding = self.heads[i](shared_features)
            class_logit = self.classifiers[i](class_embedding)
            
            embeddings.append(class_embedding)
            logits.append(class_logit)
        
        embeddings = torch.stack(embeddings, dim=1)  # [batch_size, num_classes, embedding_dim]
        logits = torch.cat(logits, dim=1)  # [batch_size, num_classes]
        
        if self.training and labels is not None:
            self.update_label_correlations(labels)
        
        return embeddings, logits