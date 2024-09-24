import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Union

class NegProtoSim(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, class_embeddings, support_y, negative_prototypes, temperature=0.07):
        # neg== 0인 샘플
        neg_indices = (support_y == 0).nonzero().squeeze()
        neg_embeddings = class_embeddings[neg_indices]
        similarities = F.cosine_similarity(neg_embeddings, negative_prototypes.unsqueeze(0), dim=1)
        scaled_similarities = similarities / temperature
        
        neg_log_likelihood = -torch.log_softmax(-scaled_similarities, dim=0)
        return neg_log_likelihood.mean()
class DistContrastive(nn.Module):
    def __init__(self, temperature=0.07, device='cuda:1') -> None:
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, querys, prototypes, labels, eps=1e-8):
        '''
        querys : [batch, embedding]
        prototypes : [embedding]
        labels : [batch] 1 is class, 0 is not class
        '''
        # Check dimensions
        assert querys.dim() == 2, f"querys should be 2D, got shape {querys.shape}"
        assert prototypes.dim() == 1, f"prototypes should be 1D, got shape {prototypes.shape}"
        assert labels.dim() == 1, f"labels should be 1D, got shape {labels.shape}"
        
        batch_size, embedding_dim = querys.shape
        assert prototypes.shape == (embedding_dim,), f"prototypes shape mismatch. Expected {(embedding_dim,)}, got {prototypes.shape}"
        assert labels.shape == (batch_size,), f"labels shape mismatch. Expected {(batch_size,)}, got {labels.shape}"

        # Normalize querys and prototypes
        querys = F.normalize(querys, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=0)

        # Compute cosine similarity
        cosine_sim = torch.matmul(querys, prototypes)  # [batch]

        # Apply temperature scaling
        logits = cosine_sim / self.temperature

        # Compute loss for each sample
        positive_mask = labels == 1
        negative_mask = labels == 0

        # Apply log-sum-exp trick
        max_logit = logits.max()
        logits_stable = logits - max_logit
        exp_logits = torch.exp(logits_stable)
        
        log_sum_exp = torch.log(exp_logits.sum() + eps) + max_logit
        
        positive_loss = -logits[positive_mask] + log_sum_exp
        negative_loss = -torch.log(1 - torch.exp(logits[negative_mask] - log_sum_exp) + eps)

        # Combine losses
        loss = torch.zeros_like(logits)
        loss[positive_mask] = positive_loss
        loss[negative_mask] = negative_loss

        # Handle potential numerical instabilities
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
        
        # Clip loss to prevent any remaining instability
        loss = torch.clamp(loss, min=0, max=50)

        # Compute average loss
        total_loss = loss.mean()

        return total_loss

    def info_nce_loss(self, features, labels, eps=1e-8):
        '''
        features : [batch, embedding]
        labels : [batch] class indices
        '''
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)

        # Compute cosine similarity
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # Discard the main diagonal from both: labels and similarities matrix
        mask = mask.fill_diagonal_(0)
        
        # Select and combine multiple positives
        positives = similarity_matrix[mask.bool()].view(batch_size, -1)

        # Select only the negatives
        negatives = similarity_matrix[~mask.bool()].view(batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        logits = logits / self.temperature

        # Apply log-sum-exp trick
        max_logit = logits.max(dim=1, keepdim=True)[0]
        logits_stable = logits - max_logit
        exp_logits = torch.exp(logits_stable)
        
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + eps) + max_logit
        
        log_prob = logits_stable - log_sum_exp

        # Compute cross entropy loss
        loss = -log_prob[torch.arange(batch_size), labels]
        
        # Handle potential numerical instabilities
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
        
        # Clip loss to prevent any remaining instability
        loss = torch.clamp(loss, min=0, max=50)

        return loss.mean()


class Contrastive(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, similarities, labels, query=False, temperature=0.07, eps=1e-8):
        # print(f"Similarities shape: {similarities.shape}")
        # print(f"Similarities range: [{similarities.min().item():.4f}, {similarities.max().item():.4f}]")
        # print(f"Labels shape: {labels.shape}")
        # print(f"Unique labels: {torch.unique(labels)}")

        # Apply temperature scaling
        similarities = similarities / temperature

        if not query:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float()

            # Exclude self-similarity
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(mask.device),
                0
            )

            # Apply log-sum-exp trick for numerical stability
            max_sim = torch.max(similarities, dim=1, keepdim=True)[0]
            exp_similarities = torch.exp(similarities - max_sim)

            # Compute positive similarities
            pos_sim = torch.sum(exp_similarities * mask * logits_mask, dim=1)

            # Compute denominator (all similarities except self-similarity)
            denom_sim = torch.sum(exp_similarities * logits_mask, dim=1)

            # print(f"Positive similarities range: [{pos_sim.min().item():.4f}, {pos_sim.max().item():.4f}]")
            # print(f"Denominator similarities range: [{denom_sim.min().item():.4f}, {denom_sim.max().item():.4f}]")

            # Compute loss
            loss = -torch.log(pos_sim / (denom_sim + eps) + eps)
        else:
            # For query mode
            exp_similarities = torch.exp(similarities)
            pos_sim = exp_similarities[torch.arange(exp_similarities.shape[0]), (1-labels).long()]
            denom_sim = torch.sum(exp_similarities, dim=1)

            # print(f"Positive similarities range: [{pos_sim.min().item():.4f}, {pos_sim.max().item():.4f}]")
            # print(f"Denominator similarities range: [{denom_sim.min().item():.4f}, {denom_sim.max().item():.4f}]")

            # Compute loss
            loss = -torch.log(pos_sim / (denom_sim + eps) + eps)

        # Handle potential numerical instabilities
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        average_loss = torch.mean(loss)
        # print(f"Average loss: {average_loss.item():.4f}")

        return average_loss
    
    

class CrossEntropy(nn.Module):
    def __init__(self, weight: Tensor = None) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C] and labels in shape [B]
        return self.criterion(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice', 'Contrastive','NegProtoSim','DistContrastive']


def get_loss(loss_fn_name: str = 'CrossEntropy', cls_weights: Union[Tensor, None] = None):
    available_loss_functions = ['CrossEntropy', 'BCEWithLogitsLoss', 'MSELoss', 'L1Loss', 'Contrastive','NegProtoSim','DistContrastive']
    
    assert loss_fn_name in available_loss_functions, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {available_loss_functions}"
    
    if loss_fn_name == 'CrossEntropy':
        return CrossEntropy(weight=cls_weights)
    elif loss_fn_name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(weight=cls_weights)
    elif loss_fn_name == 'MSELoss':
        return nn.MSELoss()
    elif loss_fn_name == 'L1Loss':
        return nn.L1Loss()
    elif loss_fn_name == 'Contrastive':
        return Contrastive()
    elif loss_fn_name == 'NegProtoSim':
        return NegProtoSim()
    elif loss_fn_name =='DistContrastive':
        return DistContrastive()


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)