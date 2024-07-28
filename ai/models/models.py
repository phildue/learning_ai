import torch
from torchvision import models as torch_models
from ai.models.vision_transformer import VisionTransformer
from ai.models.convolutional_neural_network import ConvNeuralNet

def load(model_name, shape_in, shape_out):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {
        'cnn': ConvNeuralNet(shape_out),
#        'vit': VisionTransformer((3, shape_in[0], shape_in[1]), n_patches=8, n_blocks=2, hidden_d=64, n_heads=4, mlp_dim=32, out_d=shape_out),
        'vit': VisionTransformer((3, shape_in[0], shape_in[1]), n_patches=16, n_blocks=2, hidden_d=512, n_heads=16, mlp_dim=256, out_d=shape_out),
        'vit_b_16_pretrained': torch_models.vit_b_16(weights=torch_models.ViT_B_16_Weights.DEFAULT)
    }
    model = models[model_name].to(device=device, dtype=float)
    return model
