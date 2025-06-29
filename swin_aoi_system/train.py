import argparse
import os
import yaml

from swin_aoi_system.engine.trainer import run_training


def main(args):
    """Main training script."""
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Convert dataset root path to an absolute path relative to this file
    root_dir = config["data"].get("root_dir", "dataset_root")
    if not os.path.isabs(root_dir):
        config["data"]["root_dir"] = os.path.join(os.path.dirname(__file__), root_dir)

    run_training(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AOI detection model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the training config file."
    )
    args = parser.parse_args()
    main(args)
