from .fpn import FPNHead
from .mlp import MLPHead, MLPMultiHead, Head, EfficientNetV2Head
from .embedding import MultiHeadEmbedding, MultiHeadEmbeddingBCE

__all__ = [
    'FPNHead','MLPMultiHead','Head', 'EfficientNetV2Head', 'MultiHeadEmbedding', 'MultiHeadEmbeddingBCE'
    ]