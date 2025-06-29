import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch} Loss: {avg_loss:.4f}")

from torch.utils.data import DataLoader

try:
    from torchvision.models.detection import detr_resnet50
except Exception:  # noqa: SIM105
    from torchvision.models.detection import fasterrcnn_resnet50_fpn as _frcnn

    def detr_resnet50(*, num_classes=91, pretrained=False, **kwargs):
        """Fallback to Faster R-CNN if DETR is unavailable."""
        return _frcnn(pretrained=pretrained, num_classes=num_classes, **kwargs)
from ..data.aoi_dataset import AoiDataset
from ..data.transforms import get_transforms
from ..utils.scheduler import get_scheduler
from ..utils.gpu_optimizer import apply_optimizations


def run_training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AoiDataset(
        config["data"]["root_dir"],
        data_source=config["data"].get("source", "mixed"),
        transform=get_transforms(is_train=True),
    )

    data_loader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
    )

    model = detr_resnet50(num_classes=2)
    model = apply_optimizations(model)
    model.to(device)

    optim_cfg = config.get("optimizer", {})
    opt_cls = getattr(torch.optim, optim_cfg.get("type", "AdamW"))
    optimizer = opt_cls(model.parameters(), lr=optim_cfg.get("lr", 1e-4))

    scheduler = get_scheduler(optimizer, config.get("scheduler", {}))
    num_epochs = config.get("num_epochs", 1)

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, data_loader, optimizer, device, epoch)
        scheduler.step()

    print("Training finished.")
    return model
