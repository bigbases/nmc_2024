from torch import Tensor
from nmc.models.base import BaseModel
from nmc.models.heads import MLPHead, MLPMultiHead, Head
import torch.nn as nn

class ResNet50Model(BaseModel):
    def __init__(self, backbone: str = 'ResNet-50', num_classes: int = 11):
        super().__init__(backbone, num_classes)
        self.head = Head(in_features=2048, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        _, _, _, x4 = self.backbone(x)  # Get the output from layer4
        y = self.head(x4)  # Pass through the Head
        return y

if __name__ == '__main__':
    import torch
    # model = FGMaxxVit('FGMaxxVit', 1000)
    # model.init_pretrained_fgmaxxvit('/workspace/jhmoon/nmc_2024/checkpoints/pretrained/maxvit_base_tf_512.in1k_pretrained_weights.pth')
    # x = torch.randn(2, 3, 512, 512)
    # y = model(x)
    # print(y.shape)
    model = ResNet50Model('ResNet-50', 11)
    model.init_pretrained('/workspace/jhmoon/nmc_2024/checkpoints/pretrained/resnet50_a1_in1k_weights.pth')
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    
