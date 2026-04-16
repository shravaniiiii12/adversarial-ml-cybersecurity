import torch

def feature_squeeze(images, bit_depth=4):
    levels = 2 ** bit_depth
    return torch.round(images * levels) / levels