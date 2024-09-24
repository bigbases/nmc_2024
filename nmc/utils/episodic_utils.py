import torch
from torch.nn import functional as F


def dot_similarity(embeddings, temperature=0.07, eps=1e-8):
    """
    Calculate similarity matrix using dot product with temperature scaling and numerical stability improvements.
    
    Args:
    embeddings: Tensor of shape [batch, n_class, embedding_dim]
    temperature: Float, temperature parameter for scaling
    eps: Float, small value to avoid division by zero

    Returns:
    log_similarity: Tensor of shape [n_class, batch, batch] in log space
    """
    batch_size, n_class, embedding_dim = embeddings.shape
    
    # L2 normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    
    # Permute to [n_class, batch, embedding_dim]
    embeddings = embeddings.permute(1, 0, 2)
    
    # Calculate dot product similarity
    similarity = torch.matmul(embeddings, embeddings.transpose(1, 2))
    
    # Apply temperature scaling
    similarity = similarity / temperature
    
    # Apply log-sum-exp trick for numerical stability
    max_sim = similarity.max(dim=-1, keepdim=True)[0]
    exp_sim = torch.exp(similarity - max_sim)
    
    # Compute log of normalization term
    log_normalization = torch.log(exp_sim.sum(dim=-1, keepdim=True) + eps) + max_sim
    
    # Compute log probabilities
    log_similarity = similarity - log_normalization
    
    # Clip values to prevent any remaining instability
    log_similarity = torch.clamp(log_similarity, min=-50, max=50)
    
    return log_similarity

def compute_query_similarity(query_pred, prototypes, temperature=0.07, eps=1e-8):
    # query_pred: [batch_size, n_classes, embedding_dim]
    # prototypes: [n_classes, embedding_dim]
    
    batch_size, n_classes, embedding_dim = query_pred.shape
    
    # Normalize query predictions and prototypes
    query_pred_norm = F.normalize(query_pred, p=2, dim=2)
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    
    # Compute cosine similarity
    similarities = torch.matmul(query_pred_norm, prototypes_norm.T)  # [batch_size, n_classes, n_classes]
    
    # Apply temperature scaling
    similarities = similarities / temperature
    
    # Initialize output tensor
    intra_class_similarities = torch.zeros(batch_size, n_classes, device=query_pred.device)
    
    for c in range(n_classes):
        class_similarities = similarities[:, c, c]  # [batch_size]
        
        # Apply log-sum-exp trick
        sim_max = class_similarities.max()
        sim_stable = torch.log(torch.exp(class_similarities - sim_max).mean() + eps) + sim_max
        
        intra_class_similarities[:, c] = sim_stable
    
    # Clip values to prevent any remaining instability
    intra_class_similarities = torch.clamp(intra_class_similarities, min=-50, max=50)
    
    return intra_class_similarities

def compute_prototypes_dist(embeddings, labels, temperature=0.07, eps=1e-8):
    num_classes = labels.size(1)
    batch_size, n_classes, embedding_dim = embeddings.shape
    prototypes = []
    intra_class_similarities = []
    inter_class_similarities = []

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=-1)

    for c in range(num_classes):
        positive_mask = labels[:, c] > 0
        negative_mask = labels[:, c] == 0

        if positive_mask.sum() == 0:
            dummy = torch.zeros(embedding_dim, device=embeddings.device)
            prototypes.append(dummy)
            intra_class_similarities.append(torch.tensor(float('nan'), device=embeddings.device))
            inter_class_similarities.append(torch.tensor(float('nan'), device=embeddings.device))
        else:
            positive_embeddings = embeddings[positive_mask, c, :]
            
            # Compute prototype as mean of positive embeddings
            prototype = positive_embeddings.mean(dim=0)
            prototype = F.normalize(prototype, p=2, dim=0)
            prototypes.append(prototype)

            # Compute intra-class similarity (positive pairs)
            intra_sim = torch.matmul(positive_embeddings, prototype.unsqueeze(1)).squeeze()
            intra_sim = intra_sim / temperature
            
            # Apply log-sum-exp trick
            intra_sim_max = intra_sim.max()
            intra_sim_stable = torch.log(torch.exp(intra_sim - intra_sim_max).mean() + eps) + intra_sim_max
            intra_class_similarities.append(intra_sim_stable)

            # Compute inter-class similarity (negative pairs)
            if negative_mask.sum() > 0:
                negative_embeddings = embeddings[negative_mask, c, :]
                inter_sim = torch.matmul(negative_embeddings, prototype.unsqueeze(1)).squeeze()
                inter_sim = inter_sim / temperature
                
                # Apply log-sum-exp trick
                inter_sim_max = inter_sim.max()
                inter_sim_stable = torch.log(torch.exp(inter_sim - inter_sim_max).mean() + eps) + inter_sim_max
                inter_class_similarities.append(inter_sim_stable)
            else:
                inter_class_similarities.append(torch.tensor(float('nan'), device=embeddings.device))

    prototypes = torch.stack(prototypes)
    intra_class_similarities = torch.stack(intra_class_similarities)
    inter_class_similarities = torch.stack(inter_class_similarities)

    # Clip values to prevent any remaining instability
    intra_class_similarities = torch.clamp(intra_class_similarities, max=50)
    inter_class_similarities = torch.clamp(inter_class_similarities, max=50)

    return prototypes, intra_class_similarities, inter_class_similarities



    
    
    
    
    
    
    
    
    

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
    
def calculate_negative_prototypes(support_pred, support_y):
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
    
