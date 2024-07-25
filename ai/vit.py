import torch.nn as nn
import torch
import numpy as np
from ai.positional_embedding import get_positional_embeddings
from ai.multi_headed_self_attention import MyViTBlock

class MyViT(nn.Module):
  def __init__(self, chw=(1, 28, 28), n_patches=7, hidden_d = 8, n_blocks=2, n_heads=2, out_d=10):
    # Super constructor
    super(MyViT, self).__init__()

    # Attributes
    self.chw = chw # (C, H, W)
    self.n_patches = n_patches
    self.hidden_d = hidden_d
    assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
    self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

    # 1) Linear mapper
    self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
    self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

    # 2) Learnable classifiation token
    self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

    # 3) Positional embedding
    self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(self.n_patches ** 2 + 1, self.hidden_d)))
    self.pos_embed.requires_grad = False

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(self.hidden_d, out_d),
        nn.Softmax(dim=-1)
    )

  def forward(self, images):
    patches = self.patchify(images)
    tokens = self.linear_mapper(patches)

    # Adding classification token to the tokens
    tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

    # Adding positional embedding
    pos_embed = self.pos_embed.repeat(len(tokens), 1, 1) 
    out = tokens + pos_embed #we really just add this? Yes it avoids additional dimensionality and having to learn the relations between positional and semantic data

    # Transformer Blocks
    for block in self.blocks:
        out = block(out)

    # Getting the classification token only
    out = out[:, 0]
    out = self.mlp(out) # Map to output dimension, output category distribution

    return out
  
  def patchify(self, images):
    n, c, h, w = images.shape
    n_patches = self.n_patches
    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches
  

if __name__ == '__main__':
    # Current model
    model = MyViT(
        chw=(3, 28, 28),
        n_patches=7
    )

    x = torch.randn(7, 3, 28, 28) # Dummy images
    print(model(x).shape) # torch.Size([7, 49, 16])