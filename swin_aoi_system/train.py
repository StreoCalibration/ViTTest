import argparse

def main(args):
    """Main training script."""
    print("Starting model training...")
    # 1. Load config
    # 2. Setup model, data, optimizer, loss
    # 3. Run training loop
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the AOI detection model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the training config file.')
    args = parser.parse_args()
    main(args)