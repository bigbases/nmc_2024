import torch
from torch import Tensor
from torch.nn import ModuleList
from nmc.models.base import BaseModel
from nmc.models.heads import MLPMultiHead


class FGMaxxVit_Multi(BaseModel):
    def __init__(self, backbone: str = 'FGMaxxVit_Multi_label', num_classes: int=10):
        super().__init__(backbone, num_classes)
        
        self.num_embedding= 128
        self.head = ModuleList()
        for num_class in range(num_classes):
            self.head.append(MLPMultiHead(num_features=self.backbone.config.head_hidden_size,num_embedding=self.num_embedding))
            
        self.apply(self._init_weights)
        
    def forward(self,x: Tensor, Flag = False):
        y = self.backbone(x)
        # y.shape = [batch,768]
        outputs = torch.stack([head(y,Flag) for head in self.head], dim=1)  
        # [batch, num_classes, 256]
        
        
        return outputs
        # return y
    
    
if __name__ == '__main__':
    import torch
    model = FGMaxxVit_Multi('FGMaxxVit', 10)
    # model.init_pretrained_fgmaxxvit('checkpoints/pretrained/maxvit_base_tf_512.in1k_pretrained_weights.pth')
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)  # Should output: [batch_size, num_classes]