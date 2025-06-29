# Image augmentation and transformation logic
from torchvision import transforms as T


def get_transforms(is_train=True):
    """Return basic augmentation pipeline."""
    transforms = []
    if is_train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
