import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MultiHeadedSelfAttention, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.w_q = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.w_k = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.w_v = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        N, seq_length, token_dim = sequences.shape
        d_head = token_dim // self.n_heads

        sequences = sequences.view(N, seq_length, self.n_heads, d_head)

        Q = torch.zeros(N, seq_length, self.n_heads, d_head).to(sequences.device)
        K = torch.zeros(N, seq_length, self.n_heads, d_head).to(sequences.device)
        V = torch.zeros(N, seq_length, self.n_heads, d_head).to(sequences.device)


        for head in range(self.n_heads):
            # apply different weight matrices on input data
            Q[:, :, head, :] = self.w_q[head](sequences[:, :, head, :])
            K[:, :, head, :] = self.w_k[head](sequences[:, :, head, :])
            V[:, :, head, :] = self.w_v[head](sequences[:, :, head, :])

        Q = Q.transpose(1, 2) # queries
        K = K.transpose(1, 2) # keys
        V = V.transpose(1, 2) # values

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5) # Similarity between queries and keys fuses global information
        attention_weights = self.softmax(attention_scores) # Convert similarity score to probability / weight between 0 and 1
        attention_outputs = torch.matmul(attention_weights, V) # Weight values according to global similarity
        

        attention_outputs = attention_outputs.transpose(1, 2).reshape(N, seq_length, token_dim)
        
        return attention_outputs

class VisionTransformerBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_dim):
        super(VisionTransformerBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MultiHeadedSelfAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_d)
        )

    def forward(self, x):
        out = self.norm1(x)
        out = x + self.mhsa(out)
        out = self.norm2(out)
        out = out + self.mlp(out)
        return out
    
if __name__ == '__main__':
  model = VisionTransformerBlock(hidden_d=8, n_heads=2)

  x = torch.randn(7, 50, 8)  # Dummy sequences
  print(model(x).shape)      # torch.Size([7, 50, 8])