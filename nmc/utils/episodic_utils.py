import torch
from torch.nn import functional as F


def dot_similarity(embeddings):
        #embeddings = [batch,n_class,embedding]
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        batch_size, n_class, embedding_dim = embeddings.shape
        similarity = torch.zeros(n_class, batch_size, batch_size, device=embeddings.device)
        
        for c in range(n_class):
            class_embeddings = embeddings[:, c, :]  # [batch, embedding_dim]
            similarity[c] = torch.mm(class_embeddings, class_embeddings.t())
        
        return similarity

# START ---------------Query 유사도 계산을 위한 함수 ---------------------------
def compute_prototypes_multi_label(embeddings, labels):
    num_classes = labels.size(1)
    batch_size, n_classes, embedding_dim = embeddings.shape
    prototypes = []
    for c in range(num_classes):
        positive_mask = labels[:, c] > 0  # 특정 클래스 c에 속하는 샘플을 선택
        if positive_mask.sum() == 0: 
            prototypes.append(torch.tensor([torch.zeros(embedding_dim, device=embeddings.device),torch.zeros(embedding_dim, device=embeddings.device)])) # 없으면 0 vector
        else:
            negative_mask = labels[:, c] < 1
            positive_embeddings = embeddings[positive_mask]
            negative_embeddings = embeddings[negative_mask]
            positive_prototype = positive_embeddings[:,c,:].mean(dim=0) # class 별 Embedding의 평균 추출
            negative_prototype = negative_embeddings[:,c,:].mean(dim=0) 
            prototypes.append(torch.tensor([positive_prototype,negative_prototype]))
    return torch.stack(prototypes) # ( n_class , embedding )

def dot_product_similarity(query_embeddings, prototypes):
    """
    query_embeddings: (batch_size, num_classes, embedding_dim)
    prototypes: (num_classes, embedding_dim)
    """
    similarities = torch.matmul(query_embeddings, prototypes.transpose(1, 2))  # (15, 11, 11)
    similarities = similarities.diagonal(dim1=-2, dim2=-1)  # (15, 11) 대각 행렬만 추출하여 각 class 별 embedding의 유사도를 추출 
    return similarities # ( batch size , n_class )
# END ---------------Query 유사도 계산을 위한 함수 ---------------------------
    
    
    