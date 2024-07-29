import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, dim_embedding, n_heads=2, dropout=0):
        super(MultiHeadedSelfAttention, self).__init__()
        self.dim_embedding = dim_embedding
        self.n_heads = n_heads

        assert dim_embedding % n_heads == 0, f"Can't divide dimension {dim_embedding} into {n_heads} heads"

        self.dim_head = int(dim_embedding / n_heads)
        self.w_q = nn.ModuleList([nn.Linear(self.dim_head, self.dim_head) for _ in range(self.n_heads)])
        self.w_k = nn.ModuleList([nn.Linear(self.dim_head, self.dim_head) for _ in range(self.n_heads)])
        self.w_v = nn.ModuleList([nn.Linear(self.dim_head, self.dim_head) for _ in range(self.n_heads)])
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Linear(self.dim_head*self.n_heads, self.dim_embedding)


    def forward(self, sequences):
        batch_size, seq_length, token_dim = sequences.shape

        sequences = sequences.view(batch_size, seq_length, self.n_heads, self.dim_head)

        Q = torch.zeros(batch_size, seq_length, self.n_heads, self.dim_head).to(sequences.device)
        K = torch.zeros(batch_size, seq_length, self.n_heads, self.dim_head).to(sequences.device)
        V = torch.zeros(batch_size, seq_length, self.n_heads, self.dim_head).to(sequences.device)


        for head in range(self.n_heads):
            # apply different weight matrices on input data
            Q[:, :, head, :] = self.w_q[head](sequences[:, :, head, :])
            K[:, :, head, :] = self.w_k[head](sequences[:, :, head, :])
            V[:, :, head, :] = self.w_v[head](sequences[:, :, head, :])

        Q = Q.transpose(1, 2) # queries
        K = K.transpose(1, 2) # keys
        V = V.transpose(1, 2) # values

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_head ** 0.5) # Similarity between queries and keys fuses global information
        attention_weights = self.softmax(attention_scores) # Convert similarity score to probability / weight between 0 and 1
        attention_outputs = torch.matmul(attention_weights, V) # Weight values according to global similarity
        

        attention_outputs = attention_outputs.transpose(1, 2).reshape(batch_size, seq_length, token_dim)
        out = self.mlp(attention_outputs)
        out = self.dropout(out)
        
        return out

class VisionTransformerBlock(nn.Module):
    def __init__(self, dim_embedding, n_heads, dim_mlp, dropout=0):
        super(VisionTransformerBlock, self).__init__()
        self.hidden_d = dim_embedding
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(dim_embedding)
        self.mhsa = MultiHeadedSelfAttention(dim_embedding, n_heads)
        self.norm2 = nn.LayerNorm(dim_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(dim_embedding, dim_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_mlp, dim_embedding),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = self.norm1(x)
        out = x + self.mhsa(out)
        out = self.norm2(out)
        out = out + self.mlp(out)
        return out
    
if __name__ == '__main__':
  model = VisionTransformerBlock(dim_embedding=32*32, n_heads=2, dim_mlp=256)

  x = torch.randn(7, 3, 32*32)  # Dummy sequences
  print(model(x).shape)      # torch.Size([7, 50, 8])