from torch import Tensor
from nmc.models.base import BaseModel
from nmc.models.heads import MLPHead

class FGMaxxVit(BaseModel):
    def __init__(self, backbone: str = 'FGMaxxVit', num_classes: int = 19):
        super().__init__(backbone, num_classes)
        self.head = MLPHead(self.backbone.config.head_hidden_size,num_classes)
        self.apply(self._init_weights)
        
    def forward(self,x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.head(y)
        return y
    
    
if __name__ == '__main__':
    import torch
    model = FGMaxxVit('FGMaxxVit', 1000)
    #model.init_pretrained('checkpoints/backbones/resnet/resnet18.pth')
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(y.shape)