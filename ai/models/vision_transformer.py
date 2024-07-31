import torch.nn as nn
import torch
import numpy as np
from ai.embeddings.positional import get_positional_embeddings
from ai.layers.multi_headed_self_attention import VisionTransformerBlock


class VisionTransformer(nn.Module):
  def __init__(self, chw=(1, 28, 28), n_patches=7, hidden_d = 8, n_blocks=2, n_heads=2, dim_out=10, mlp_dim=64):
    # Super constructor
    super(VisionTransformer, self).__init__()

    # Attributes
    self.chw = chw # (C, H, W)
    self.n_patches = n_patches
    self.dim_latent = hidden_d
    assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

    # 1) Linear mapper
    self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
    self.linear_mapper = nn.Linear(self.input_d, self.dim_latent)

    # 2) Learnable classifiation token
    self.class_token = nn.Parameter(torch.rand(1, self.dim_latent))

    # 3) Positional embedding
    self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.dim_latent)))
    self.pos_embed.requires_grad = False

    # 4) Transformer encoder blocks
    self.transformer_blocks = nn.Sequential(*[VisionTransformerBlock(hidden_d, n_heads, mlp_dim) for _ in range(n_blocks)])

    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(self.dim_latent, dim_out),
        nn.Softmax(dim=-1)
    )

  def forward(self, images):
    

    # Eq (1)
    E = self.patchify(images)
    x_pE = self.linear_mapper(E)
    E_pos = self.pos_embed.repeat(x_pE.shape[0], 1, 1) 

    z_0 = torch.stack([torch.vstack((self.class_token, x_pE[i])) for i in range(len(x_pE))]) + E_pos

    
    out = self.transformer_blocks(z_0)

    # Getting the classification token only
    out = out[:, 0]
    out = self.mlp(out) # Map to output dimension, output category distribution

    return out
  
  def patchify(self, images):
    B, C, H, W = images.shape
    N = self.n_patches
    assert H == W, "Patchify method is implemented for square images only"

    P = H // N
    # See ViT 3.1 First paragraph
    x_p = images.unfold(2, P, P).unfold(3, P, P)
    x_p = x_p.permute(0, 2, 3, 1, 4, 5).contiguous()
    x_p = x_p.view(B, N * N, C * P * P)

    return x_p

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    # Current model
    model = VisionTransformer(
        chw=(3, 28, 28),
        n_patches=7
    )

    x = torch.randn(7, 3, 28, 28) # Dummy images
    print(model(x).shape) # torch.Size([7, 49, 16])