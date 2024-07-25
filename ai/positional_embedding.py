import torch
import numpy as np

# Vaswani et al. (https://arxiv.org/abs/1706.03762)
def get_positional_embeddings(sequence_length, d):
    """Returns positional embedding based on sequence length (number of tokens) and d (dimensionality) of each token.
    """
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


if __name__ == "__main__":
  import matplotlib.pyplot as plt

  plt.imshow(get_positional_embeddings(100, 300), cmap="hot", interpolation="nearest")
  plt.show()