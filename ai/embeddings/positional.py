import torch
import numpy as np

# Vaswani et al. (https://arxiv.org/abs/1706.03762)
def get_positional_embeddings(sequence_length, d):
    """Returns positional embedding based on sequence length (number of tokens) and d (dimensionality) of each token.
    """
    i = np.arange(sequence_length)[:, np.newaxis]
    j = np.arange(d)
    exponent = j / d
    m = 10000
    denominator = m ** exponent
    
    return np.where(j % 2 == 0, np.sin(i / denominator), np.cos(i / (m ** ((j - 1) / d))))
    

if __name__ == "__main__":
  import matplotlib.pyplot as plt

  plt.imshow(get_positional_embeddings(100, 300), cmap="hot", interpolation="nearest")
  plt.show()