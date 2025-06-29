import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch


class AoiDataset(Dataset):
    """Dataset for real and synthetic AOI images."""

    def __init__(self, root_dir, data_source="mixed", transform=None):
        """Collect file paths for the chosen data source.

        Parameters
        ----------
        root_dir : str
            Path to ``dataset_root`` directory.
        data_source : str
            ``"real"``, ``"synthetic"`` or ``"mixed"``.
        transform : callable, optional
            Optional transform applied to PIL images.
        """

        self.root_dir = root_dir
        self.data_source = data_source
        self.transform = transform
        self.samples = []

        if data_source in ("synthetic", "mixed"):
            img_dir = os.path.join(root_dir, "synthetic_data", "images")
            ann_dir = os.path.join(root_dir, "synthetic_data", "annotations")
            for ann_file in sorted(os.listdir(ann_dir)):
                if not ann_file.endswith(".json"):
                    continue
                with open(os.path.join(ann_dir, ann_file), "r", encoding="utf-8") as f:
                    ann = json.load(f)
                img_path = os.path.join(img_dir, ann["image_name"])
                boxes = [d["bbox"] for d in ann.get("defects", [])]
                labels = [1] * len(boxes)
                self.samples.append((img_path, boxes, labels))

        if data_source in ("real", "mixed"):
            real_dir = os.path.join(root_dir, "real_data")
            for file in sorted(os.listdir(real_dir)):
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    img_path = os.path.join(real_dir, file)
                    self.samples.append((img_path, [], []))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes, labels = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        if self.transform:
            image = self.transform(image)

        return image, target
