import torch
from torch import Tensor
from nmc.models.base import BaseModel
from nmc.models.heads import EfficientNetV2Head

class EfficientNetV2MModel(BaseModel):
    def __init__(self, backbone: str = 'EfficientNetV2', num_classes: int = 11):
        super().__init__(backbone, num_classes)
        self.head = EfficientNetV2Head(in_features=self.backbone.channels[-1], num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)  # Get the final feature vector from the backbone
        y = self.head(x)      # Pass through the Head
        return y  # Return logits


if __name__ == '__main__':
    model = EfficientNetV2MModel('EfficientNetV2', 8)
    model.init_pretrained('/workspace/jhmoon/nmc_2024/checkpoints/pretrained/tf_efficientnetv2_m_weights.pth')
    for name, param in model.named_parameters():
        print(param[0][0])
        break
    # # print(y)
    # x = torch.randn(8, 3, 512, 512)
    # features = model(x)
    # print(features.shape)  # Should output (8, 1280)
