# Image augmentation and transformation logic
import random
from torchvision import transforms as T


class Compose:
    """
    이미지와 타겟(바운딩 박스 등)을 함께 변환하는 Compose 클래스.
    torchvision의 기본 Compose는 타겟을 처리하지 못하므로 재정의합니다.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip:
    """
    주어진 확률로 이미지와 바운딩 박스를 수평으로 뒤집습니다.
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = T.functional.hflip(image)
            if "boxes" in target:
                bbox = target["boxes"]
                # PIL Image.size는 (width, height)를 반환합니다.
                width, _ = image.size
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
        return image, target

class ToTensor:
    """
    이미지를 텐서로 변환합니다. 타겟은 그대로 반환합니다.
    """
    def __call__(self, image, target):
        return T.functional.to_tensor(image), target

def get_transforms(is_train):
    """is_train 플래그에 따라 적절한 변환 파이프라인을 반환합니다."""
    transforms = []
    if is_train:
        # 학습 시에만 좌우 반전 증강을 적용합니다.
        transforms.append(RandomHorizontalFlip(0.5))
    # 모든 경우에 이미지를 텐서로 변환합니다.
    transforms.append(ToTensor())
    return Compose(transforms)
