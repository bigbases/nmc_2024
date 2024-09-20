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
    def __init__(self, scale_factor=10.0, device='cuda:1') -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.bn = nn.BatchNorm1d(1).to(device)
        self.device = device
    def forward(self, class_distances, class_labels, pos_margin=0.1, neg_margin=0.9):
        # 거리 스케일 조정 및 배치 정규화 적용
        class_distances = self.bn(class_distances.unsqueeze(1)).squeeze(1)
        class_distances = self.scale_factor * class_distances
        
        # 거리를 유사도로 변환 (선택적)
        # similarity = 1 / (1 + class_distances)
        # class_distances = 1 - similarity  # 유사도가 높을수록 거리가 작아지도록

        positive_mask = class_labels == 1
        negative_mask = class_labels == 0
        
        # L1 손실(절대값) 사용
        positive_loss = F.relu(class_distances - pos_margin).abs() * positive_mask.float()
        negative_loss = F.relu(neg_margin - class_distances).abs() * negative_mask.float()
        
        num_positives = positive_mask.sum()
        num_negatives = negative_mask.sum()
        
        if num_positives > 0:
            positive_loss = positive_loss.sum() / num_positives
        else:
            positive_loss = torch.tensor(0., device=class_distances.device)
        
        if num_negatives > 0:
            negative_loss = negative_loss.sum() / num_negatives
        else:
            negative_loss = torch.tensor(0., device=class_distances.device)
        
        # 로그 스케일 사용 및 엡실론 추가
        total_loss = torch.log1p(positive_loss + negative_loss + 1e-6)
        
        return total_loss
class Contrastive(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,similarities, labels, query = False, temperature=0.07, eps=1e-8):
        # similarities : [batch,batch] 특정 class embedding의 similarity matrix
        # labels : [batch] 특정 class embedding의 class 정보
        
        # temperature(T) scaling
        similarities = similarities / temperature
        # exp
        exp_similarities = torch.exp(similarities)
        
        if query == False:
            # class mask : [batch,batch]
            labels = labels.contiguous().view(-1, 1)          
            mask = torch.eq(labels, labels.T).float()

            # Exclude self-similarity
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(mask.device),
                0
            )
            pos_sim = (mask * exp_similarities * logits_mask)
            pos_neg_sim = (logits_mask * exp_similarities).sum(1, keepdim=True)
        else:
            pos_sim = exp_similarities[torch.arange(exp_similarities.shape[0]),(1-labels).long()]
            pos_neg_sim = exp_similarities.sum(dim=1)
        
        loss_matrix = -torch.log((pos_sim + eps)/(pos_neg_sim+ eps))
        fit_mask = torch.isfinite(loss_matrix)
        fit_matrix = torch.where(fit_mask, loss_matrix, torch.zeros_like(loss_matrix))
        total_loss = fit_matrix.sum()
        valid_count = fit_mask.sum()

        average_loss = total_loss / valid_count
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