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
def compute_prototypes_multi_label(embeddings, labels, num_classes):
    batch_size, n_classes, embedding_dim = embeddings.shape
    prototypes = []
    for c in range(num_classes):
        class_mask = labels[:, c] > 0  # 특정 클래스 c에 속하는 샘플을 선택
        if class_mask.sum() == 0:
            prototypes.append(torch.zeros(embedding_dim, device=embeddings.device)) # 없으면 0 vector
        else:
            class_embeddings = embeddings[class_mask]
            prototype = class_embeddings[:,c,:].mean(dim=0) # class 별 Embedding의 평균 추출 
            prototypes.append(prototype)
    return torch.stack(prototypes) # ( n_class , embedding )

def dot_product_similarity(query_embeddings, prototypes):
    """
    query_embeddings: (batch_size, num_classes, embedding_dim)
    prototypes: (1, num_classes, embedding_dim)
    """
    similarities = torch.matmul(query_embeddings, prototypes.transpose(1, 2))  # (15, 11, 11)
    similarities = similarities.diagonal(dim1=-2, dim2=-1)  # (15, 11) 대각 행렬만 추출하여 각 class 별 embedding의 유사도를 추출 
    return similarities # ( batch size , n_class )
# END ---------------Query 유사도 계산을 위한 함수 ---------------------------
    
    
    