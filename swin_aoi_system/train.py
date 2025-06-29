import argparse
import yaml

from .engine.trainer import run_training


def main(args):
    """Main training script."""
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_training(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AOI detection model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the training config file."
    )
    args = parser.parse_args()
    main(args)