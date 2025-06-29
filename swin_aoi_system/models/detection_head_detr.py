import torch.nn as nn

class DetrDetectionHead(nn.Module):
    """
    DETR-style detection head with Transformer Decoder and FFN.
    """
    def __init__(self):
        super().__init__()