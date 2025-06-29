import torch.nn as nn
from torchvision.models.detection import detr_resnet50


class DetrDetectionHead(nn.Module):
    """Simple wrapper around torchvision's DETR model."""

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.model = detr_resnet50(pretrained=pretrained)
        in_features = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

