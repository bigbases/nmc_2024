from torch import Tensor
from torch.nn import ModuleList
from nmc.models.base import BaseModel
from nmc.models.heads import MLPMultiHead


class FGMaxxVit_Multi(BaseModel):
    def __init__(self, backbone: str = 'FGMaxxVit', num_classes: int=10):
        super().__init__(backbone, num_classes)
        
        
        self.head = ModuleList()
        for num_class in range(num_classes):
            self.head.append(MLPMultiHead(num_features=self.backbone.config.head_hidden_size,num_embedding=256))
            
        self.apply(self._init_weights)
        
    def forward(self,x: Tensor, task_type : int):
        y = self.backbone(x)
        
        # y = self.head(y)
        return self.head[task_type](y)
        # return y
    
    
if __name__ == '__main__':
    import torch
    model = FGMaxxVit_Multi('FGMaxxVit', 10)
    model.init_pretrained_fgmaxxvit('checkpoints/pretrained/maxvit_base_tf_512.in1k_pretrained_weights.pth')
    x = torch.randn(2, 3, 512, 512)
    y = model(x,0)
    print(y.shape)