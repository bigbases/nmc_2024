import torch
from torch.nn import functional as F


def dot_similarity(embeddings):
        #embeddings = [batch,n_class,embedding]
        batch_size, n_class, embedding_dim = embeddings.shape
        embeddings = embeddings.permute(1, 0, 2)
        # Calculate cosine similarity
        similarity = F.cosine_similarity(embeddings.unsqueeze(2), embeddings.unsqueeze(1), dim=-1)
    
        return similarity

# START ---------------Query 유사도 계산을 위한 함수 ---------------------------
def compute_prototypes_multi_label(embeddings, labels):
    num_classes = labels.size(1)
    batch_size, n_classes, embedding_dim = embeddings.shape
    prototypes = []
    for c in range(num_classes):
        positive_mask = labels[:, c] > 0  # 특정 클래스 c에 속하는 샘플을 선택
        if positive_mask.sum() == 0:
            dumy = torch.zeros(embedding_dim, device=embeddings.device)
            prototypes.append(torch.stack([dumy,dumy],dim=0)) # 없으면 0 vector
        else:
            negative_mask = labels[:, c] < 1
            positive_embeddings = embeddings[positive_mask]
            negative_embeddings = embeddings[negative_mask]
            positive_prototype = positive_embeddings[:,c,:].mean(dim=0) # class 별 Embedding의 평균 추출
            negative_prototype = negative_embeddings[:,c,:].mean(dim=0)
            
            combined_prototypes = torch.stack([positive_prototype, negative_prototype], dim=0)

            prototypes.append(combined_prototypes)
    return torch.stack(prototypes) # ( n_class , embedding )

def dot_product_similarity(query_embeddings, prototypes):
    """
    query_embeddings: (batch_size, num_classes, embedding_dim)
    prototypes: (num_classes, pn, embedding_dim)
    """
    query_embeddings = query_embeddings.unsqueeze(2)
    prototypes = prototypes.unsqueeze(0)
    
    similarities = F.cosine_similarity(query_embeddings, prototypes, dim=-1)
    
    return similarities
    
def  (support_pred, support_y):
    batch_size, n_class, embedding_dim = support_pred.shape
    
    negative_prototypes = []
    class_exists = (support_y.sum(dim=0) > 0)
    
    for i in range(n_class):
    # 중요!! cls learning을 수행할 수 있을 때만 negative sample 수집
    if class_exists[i]:
        zero_indices = (support_y[:, i] == 0).nonzero().squeeze()
        
        if zero_indices.dim() > 0:
            negative_prototypes.append(support_pred[:, i, :][zero_indices])
        
    if len(negative_prototypes) >0:
        negative_prototypes = torch.cat(negative_prototypes, dim=0)
        return negative_prototypes.mean(dim=0)
        
    else:
        return None
    
