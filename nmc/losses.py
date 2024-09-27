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
    def __init__(self, temperature=0.07, margin=0.5, device='cuda:1') -> None:
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.device = device

    def forward(self, querys, prototypes, labels, eps=1e-8):
        '''
        querys : [batch, embedding]
        prototypes : [embedding]
        labels : [batch] 1 is class, 0 is not class
        '''
        # 차원 체크 (이전과 동일)
        assert querys.dim() == 2, f"querys should be 2D, got shape {querys.shape}"
        assert prototypes.dim() == 1, f"prototypes should be 1D, got shape {prototypes.shape}"
        assert labels.dim() == 1, f"labels should be 1D, got shape {labels.shape}"
        
        batch_size, embedding_dim = querys.shape
        assert prototypes.shape == (embedding_dim,), f"prototypes shape mismatch. Expected {(embedding_dim,)}, got {prototypes.shape}"
        assert labels.shape == (batch_size,), f"labels shape mismatch. Expected {(batch_size,)}, got {labels.shape}"

        # 쿼리와 프로토타입 정규화
        querys = F.normalize(querys, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=0)

        # 코사인 유사도 계산
        cosine_sim = torch.matmul(querys, prototypes)  # [batch]

        # 온도 스케일링 적용
        logits = cosine_sim / self.temperature

        # 양성과 음성 샘플 분리
        positive_mask = labels == 1
        negative_mask = labels == 0

        # Compute normalized probabilities
        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum() + 1e-8

        # Positive loss (attraction)
        positive_probs = exp_logits[positive_mask] / sum_exp_logits
        positive_loss = -torch.log(positive_probs + 1e-8).mean()

        # Negative loss (repulsion with margin)
        negative_probs = exp_logits[negative_mask] / sum_exp_logits
        negative_loss = torch.clamp(negative_probs - self.margin, min=0).mean()
        # Combine losses
        total_loss = positive_loss + negative_loss

        return total_loss

class Contrastive(nn.Module):
        def __init__(self, temperature=0.07):
            super().__init__()
            self.temperature = temperature

        def forward(self, similarities, labels):
            device = similarities.device
            batch_size = similarities.shape[0]

            # Create a mask for positive pairs
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
            
            # Remove self-similarities from positive mask
            pos_mask.fill_diagonal_(0)
            
            # For numerical stability
            logits = similarities / self.temperature
            
            # Compute log_prob
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            
            # Compute mean of log-likelihood over positive pairs
            mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
            
            # Loss
            loss = -mean_log_prob_pos.mean()

            return loss


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
        return nn.BCEWithLogitsLoss(weight=cls_weights,reduction='none')
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