from torch import nn, Tensor
from nmc.models.base import BaseModel
import torch

class MultiClassHead(nn.Module):
    """A multi-class head with a fully connected layer for each class."""
    def __init__(self, in_features: int, num_classes: int, embedding_size: int = 128):
        super(MultiClassHead, self).__init__()
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

class ResNet50MultiHeadModel(BaseModel):
    def __init__(self, backbone: str = 'ResNet-50', num_classes: int = 11, embedding_size: int = 128):
        super().__init__(backbone, num_classes)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling layer
        self.head = MultiClassHead(in_features=2048, num_classes=num_classes, embedding_size=embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        _, _, _, x4 = self.backbone(x)          # Get the output from layer4 (encoder output)
        x4 = self.global_avg_pool(x4)           # Apply global average pooling
        x4 = x4.view(x4.size(0), -1)            # Flatten to (batch_size, 2048)
        y = self.head(x4)                       # Pass through the multi-class head
        return y  # Shape: [batch_size, num_classes]

# Example usage
if __name__ == '__main__':
    model = ResNet50MultiHeadModel('ResNet-50', num_classes=11, embedding_size=128)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)  # Should output: [batch_size, num_classes]
