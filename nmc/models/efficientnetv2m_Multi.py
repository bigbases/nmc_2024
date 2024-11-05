from torch import nn, Tensor
from nmc.models.base import BaseModel
import torch

class EfficientNetV2MultiClassHead(nn.Module):
    """Multi-class head with a separate fully connected layer for each class."""
    def __init__(self, in_features: int = 1280, num_classes: int = 11, embedding_size: int = 128):
        super(EfficientNetV2MultiClassHead, self).__init__()
        self.classifier_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, 1)
            ) for _ in range(num_classes)
        ])

    def forward(self, x: Tensor) -> Tensor:
        # Apply each classifier head independently to the encoder output
        return torch.cat([head(x) for head in self.classifier_heads], dim=1)

class EfficientNetV2MModelMulti(BaseModel):
    def __init__(self, backbone: str = 'EfficientNetV2', num_classes: int = 11, embedding_size: int = 128):
        super().__init__(backbone, num_classes)
        # Multi-class head with independent classifiers for each class
        self.head = EfficientNetV2MultiClassHead(in_features=self.backbone.channels[-1], num_classes=num_classes, embedding_size=embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)  # Get the final feature vector from the backbone
        y = self.head(x)      # Pass through each classifier head independently
        return y  # Shape: [batch_size, num_classes]

# Example usage
if __name__ == '__main__':
    model = EfficientNetV2MModelMulti('EfficientNetV2', num_classes=11, embedding_size=128)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)  # Expected output: [batch_size, num_classes]
