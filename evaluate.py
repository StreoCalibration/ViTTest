import argparse
import torch
from tqdm import tqdm
from torchvision.ops import box_iou

from engine.trainer import get_model_instance, collate_fn
from engine.predictor import predict
from data.aoi_dataset import AoiDataset
from data.transforms import get_transforms


def evaluate(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    """Simple evaluation loop returning precision, recall and F1-score."""
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for images, targets in tqdm(data_loader, desc="Evaluating"):
        outputs = predict(model, images, device)
        for output, target in zip(outputs, targets):
            gt_boxes = target["boxes"].cpu()
            pred_boxes = output["boxes"].detach().cpu()
            scores = output.get("scores", torch.ones(len(pred_boxes))).cpu()

            pred_boxes = pred_boxes[scores > score_threshold]

            if gt_boxes.numel() == 0:
                false_positive += len(pred_boxes)
                continue

            if pred_boxes.numel() == 0:
                false_negative += len(gt_boxes)
                continue

            ious = box_iou(pred_boxes, gt_boxes)
            matched_gt = set()
            for i in range(pred_boxes.size(0)):
                j = torch.argmax(ious[i]).item()
                if ious[i, j] >= iou_threshold and j not in matched_gt:
                    true_positive += 1
                    matched_gt.add(j)
                else:
                    false_positive += 1
            false_negative += gt_boxes.size(0) - len(matched_gt)

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1_score


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AoiDataset(
        root_dir=args.data_root,
        data_source=args.data_source,
        transform=get_transforms(is_train=False),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    model = get_model_instance(num_classes=2)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    precision, recall, f1 = evaluate(
        model,
        data_loader,
        device,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
    )

    print("\n평가 결과")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the AOI detection model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file (.pth).")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument(
        "--data-source",
        type=str,
        default="synthetic",
        choices=["synthetic", "real", "mixed"],
        help="Which subset of data to use for evaluation.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold to determine true positives.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Confidence score threshold for predictions.",
    )
    args = parser.parse_args()
    main(args)
