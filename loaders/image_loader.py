# loaders/image_loader.py

from PIL import Image

def load_image(file_path: str) -> list[Image.Image]:
    """
    Load a standalone image file using PIL.
    Returns a list containing the PIL Image object.
    """
    image = Image.open(file_path).convert("RGB")
    return [image]