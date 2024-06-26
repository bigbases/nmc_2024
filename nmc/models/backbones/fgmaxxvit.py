from torch import nn
from timm.models import MaxxVit, MaxxVitCfg, MaxxVitConvCfg, MaxxVitTransformerCfg

class FGMaxxVit(nn.Module):
    def __init__(self, img_size=512, num_classes=1000):
        super().__init__()
        self.config = self._get_config()
        self.encoder = MaxxVit(cfg=self.config, img_size=img_size)
        self.stem = self.encoder.stem
        self.stages = self.encoder.stages

    def _get_config(self):
        return MaxxVitCfg(
            embed_dim=(96, 192, 384, 768),
            depths=(2, 6, 14, 2),
            block_type=('M',) * 4,
            stem_width=64,
            stem_bias=True,
            head_hidden_size=768,
            **self._tf_cfg()
        )
    def _tf_cfg(self):
        return dict(
            conv_cfg=MaxxVitConvCfg(
                norm_eps=1e-3,
                act_layer='gelu_tanh',
                padding='same',
            ),
            transformer_cfg=MaxxVitTransformerCfg(
                norm_eps=1e-5,
                act_layer='gelu_tanh',
                head_first=False,
                rel_pos_type='bias_tf',
            ),
        )

    def forward(self, x):
        x = self.stem(x)
        features = self.stages(x)
        return features
    
    
if __name__ == '__main__':
    import torch
    model = FGMaxxVit(img_size=512, num_classes=1000)
    x = torch.zeros(1, 3, 512, 512)
    outs = model(x)
    print(outs.shape)