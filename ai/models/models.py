import torch
from ai.models.vision_transformer import VisionTransformer
from ai.models.convolutional_neural_network import ConvNeuralNet

def load(model_name, shape_in, shape_out):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {
        'cnn': ConvNeuralNet(shape_out),
        'vit': VisionTransformer((3, shape_in[0], shape_in[1]), n_patches=8, n_blocks=2, hidden_d=256, n_heads=4, mlp_dim=128, out_d=shape_out)
    }
    model = models[model_name].to(device=device, dtype=float)
    return model
