import torch
from torch import nn, Tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride, padding)

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.proj(x)                   # b x hidden_dim x 14 x 14
        return x


class Pooling(nn.Module):
    def __init__(self, pool_size=3) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, 1, pool_size//2, count_include_pad=False)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x) - x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))

class DecBlk(nn.Module):
    def __init__(self, dim, pool_size=3, dpr=0., layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = nn.GroupNorm(1, dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.mlp = MLP(dim, dim)
        
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x))) 
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))) 
        return x

class PoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3, dpr=0., layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = Pooling(pool_size)
        self.norm2 = nn.GroupNorm(1, dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.mlp = MLP(dim, int(dim*4))
        
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x))) 
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))) 
        return x

poolformer_settings = {
    'S12': [[2, 2, 6, 2], [64, 128, 320, 512], 0.],   
    'S24': [[4, 4, 12, 4], [64, 128, 320, 512], 0.1],       # [layers, embed_dims, drop_path_rate]
    'S36': [[6, 6, 18, 6], [64, 128, 320, 512], 0.2],
    'M36': [[6, 6, 18, 6], [96, 192, 384, 768], 0.3]
}


class PoolFormer(nn.Module):     
    def __init__(self, model_name: str = 'S24') -> None:
        super().__init__()
        assert model_name in poolformer_settings.keys(), f"PoolFormer model name should be in {list(poolformer_settings.keys())}"
        layers, embed_dims, drop_path_rate = poolformer_settings[model_name]
        self.channels = embed_dims
    
        self.patch_embed = PatchEmbed(7, 4, 2, 3, embed_dims[0])

        network = []

        for i in range(len(layers)):
            blocks = []
            for j in range(layers[i]):
                dpr = drop_path_rate * (j + sum(layers[:i])) / (sum(layers) - 1)
                blocks.append(PoolFormerBlock(embed_dims[i], 3, dpr))

            network.append(nn.Sequential(*blocks))
            if i >= len(layers) - 1: break
            network.append(PatchEmbed(3, 2, 1, embed_dims[i], embed_dims[i+1]))

        self.network = nn.ModuleList(network)

        self.out_indices = [0, 2, 4, 6]
        for i, index in enumerate(self.out_indices):
            self.add_module(f"norm{index}", nn.GroupNorm(1, embed_dims[i]))

    def forward(self, x: Tensor):
        x = self.patch_embed(x)
        outs = []

        for i, blk in enumerate(self.network):
            x = blk(x)

            if i in self.out_indices:
                out = getattr(self, f"norm{i}")(x)
                outs.append(out)
        return outs
        

if __name__ == '__main__':
    model = PoolFormer('S12').cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    feats = model(x)
    for y in feats:
        print(y.shape)
"""
torch.Size([2, 64, 128, 128])
torch.Size([2, 128, 64, 64])
torch.Size([2, 320, 32, 32])
torch.Size([2, 512, 16, 16])
"""