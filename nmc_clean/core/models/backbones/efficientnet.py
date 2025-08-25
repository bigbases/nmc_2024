from torch import nn, Tensor
import torch.nn.functional as F
from timm import create_model

class EfficientNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the EfficientNetV2 model with pretrained weights
        self.encoder = create_model('tf_efficientnetv2_m.in21k_ft_in1k', pretrained=True)
        
        # Access stem, stage layers, and conv_head directly from the model
        self.stem = self.encoder.conv_stem
        self.stages = nn.Sequential(
            self.encoder.blocks[:3],   # Stages 1 to 3 (FusedMBConv layers)
            self.encoder.blocks[3:],   # Stages 4 to 7 (MBConv layers)
        )
        self.conv_head = self.encoder.conv_head
        self.bn2 = self.encoder.bn2
        self.channels = [24, 48, 80, 160, 176, 304, 512, 1280]

    def forward(self, x: Tensor) -> Tensor:
        # Forward pass through the stem, stages, conv_head, and bn2 layers
        x = self.stem(x)
        x = self.stages(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        return x.view(x.size(0), -1)  # Flatten to shape (batch_size, 1280)

if __name__ == '__main__':
    import torch
    model = EfficientNetV2()
    x = torch.randn(8, 3, 512, 512)
    features = model(x)
    print(features.shape)  # Should output (8, 1280)
