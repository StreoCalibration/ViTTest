import torch.nn as nn

class ScaledCosineAttention(nn.Module):
    def __init__(self):
        super().__init__()