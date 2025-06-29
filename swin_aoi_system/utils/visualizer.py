# Utilities for visualizing model predictions
from PIL import ImageDraw


def draw_predictions(image, predictions):
    """Draw bounding boxes on the given PIL Image."""
    draw = ImageDraw.Draw(image)
    for box in predictions:
        draw.rectangle(list(box), outline="red", width=2)
    return image
