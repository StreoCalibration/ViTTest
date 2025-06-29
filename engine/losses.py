import torch.nn as nn

class HungarianMatcherLoss(nn.Module):
    """
    Implements the bipartite matching loss from DETR.
    This class combines the Hungarian algorithm for matching predictions
    to ground truth targets and the final loss calculation.
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()