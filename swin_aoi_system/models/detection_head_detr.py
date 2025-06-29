import torch.nn as nn

try:
    from torchvision.models.detection import detr_resnet50
except Exception:  # noqa: SIM105
    from torchvision.models.detection import fasterrcnn_resnet50_fpn as _frcnn

    def detr_resnet50(*, num_classes=91, pretrained=False, **kwargs):
        """Fallback to Faster R-CNN if DETR is unavailable."""
        return _frcnn(pretrained=pretrained, num_classes=num_classes, **kwargs)


class DetrDetectionHead(nn.Module):
    """Simple wrapper around torchvision's DETR model."""

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.model = detr_resnet50(pretrained=pretrained)
        in_features = self.model.class_embed.in_features
        self.model.class_embed = nn.Linear(in_features, num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)

