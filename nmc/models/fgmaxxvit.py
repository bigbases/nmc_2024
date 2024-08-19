from torch import Tensor
from nmc.models.base import BaseModel
from nmc.models.heads import MLPHead, MLPMultiHead
import torch.nn as nn


class FGMaxxVit(BaseModel):
    def __init__(self, backbone: str = 'FGMaxxVit', num_classes: int = 11):
        super().__init__(backbone, num_classes)
        self.head = MLPHead(self.backbone.config.head_hidden_size,num_classes)
        self.apply(self._init_weights)
        
    def forward(self,x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.head(y)
        return y
    
class Multi_FGMaxxVit(BaseModel):
    def __init__(self, backbone: str = 'FGMaxxVit', num_classes: int = 11):
        super().__init__(backbone, num_classes)
        self.heads = nn.ModuleList([MLPMultiHead(self.backbone.config.head_hidden_size) for _ in range(num_classes)])
        self.num_classes = num_classes
        self.apply(self._init_weights)
        
    def forward(self, x: Tensor, class_indices: Tensor) -> Tensor:
        features = self.backbone(x)
        logits_list = []
        features_list = []

        for i in range(self.num_classes):
            if i in class_indices:
                logits, feature = self.heads[i](features)
                logits_list.append(logits)
                features_list.append(feature)
            else:
                logits_list.append(torch.zeros(x.size(0), 1, device=x.device)) 
                features_list.append(torch.zeros(x.size(0), self.backbone.config.head_hidden_size, device=x.device))

        logits = torch.cat(logits_list, dim=1)  # Concatenate outputs from all heads
        return logits, features_list

    def compute_loss(self, logits: Tensor, targets: Tensor, class_indices: Tensor, criterion) -> Tensor:
        loss = 0.0
        for i in range(self.num_classes):
            if i in class_indices:
                loss += criterion(logits[:, i], targets[:, i])
        return loss
    
if __name__ == '__main__':
    import torch
    model = FGMaxxVit('FGMaxxVit', 1000)
    model.init_pretrained_fgmaxxvit('checkpoints/pretrained/maxvit_base_tf_512.in1k_pretrained_weights.pth')
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(y.shape)