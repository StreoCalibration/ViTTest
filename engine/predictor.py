import torch


@torch.no_grad()
def predict(model, images, device=None):
    """Run inference on a single image or a list of images.

    Args:
        model: Trained PyTorch model.
        images: ``torch.Tensor`` or list of tensors with shape [C, H, W].
        device: Optional device to run inference on. If ``None`` the device
            of the model parameters is used.

    Returns:
        List of prediction dictionaries as returned by ``model``.
    """
    model.eval()

    if not isinstance(images, (list, tuple)):
        images = [images]

    device = device or next(model.parameters()).device
    batch = [img.to(device) for img in images]
    outputs = model(batch)
    return outputs
