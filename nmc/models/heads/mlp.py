from torch import nn, Tensor
from timm.layers import SelectAdaptivePool2d, LayerNorm2d

class MLPHead(nn.Module):
    def __init__(self, num_features, num_classes, pool_type='avg', drop_rate=0.0):
        super().__init__()
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=False)
        self.norm = LayerNorm2d(num_features, eps=1e-5)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.pre_logits = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        logits = self.fc(x)
        return logits

class MLPMultiHead(nn.Module):
    def __init__(self, num_features, pool_type='avg', drop_rate=0.0):
        super().__init__()
        self.global_pool = SelectAdaptivePool2d(pool_type=pool_type, flatten=False)
        self.norm = LayerNorm2d(num_features, eps=1e-5)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.pre_logits_linear = nn.Linear(num_features, num_features)
        self.pre_logits_activation = nn.Tanh()  
        self.drop = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()  # Sigmoid for multi-label classification

    def forward(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        feature = self.pre_logits_linear(x)  # Class-specific representation before activation
        x = self.pre_logits_activation(feature)  
        x = self.drop(x)
        logits = self.fc(x)
        logits = self.sigmoid(logits)  # Apply sigmoid activation
        return logits, feature
    
    
if __name__ == '__main__':
    import torch
    head = MLPHead(num_features=768, num_classes=1000)
    features = torch.zeros(1, 768, 16, 16)
    outs = head(features)
    print(outs.shape)
    
