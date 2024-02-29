import json
from pathlib import Path
from typing import List, Literal, Tuple

import torch
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor
from utils import get_device, load_params


def load_model(
    pretrained_model_name_or_path: str, cache_dir: str
) -> Tuple[CLIPModel, CLIPProcessor]:
    """Load model and processor from HF."""
    model = CLIPModel.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        cache_dir=cache_dir,
    )

    processor = CLIPProcessor.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        cache_dir=cache_dir,
    )

    return model.eval(), processor

def get_text_embeddings(
    text_input: str | List[str], model: CLIPModel, processor: CLIPProcessor
) -> torch.Tensor:
    inputs = processor(text=text_input, return_tensors="pt", padding=True)
    inputs.to(model.device)
    with torch.no_grad():
        results = model.get_text_features(**inputs)
    return results


def get_image_embeddings(
    image_input: Image.Image | List[Image.Image],
    model: CLIPModel,
    processor: CLIPProcessor,
) -> torch.Tensor:
    inputs = processor(images=image_input, return_tensors="pt")
    inputs.to(model.device)
    with torch.no_grad():
        results = model.get_image_features(**inputs)
    return results


if __name__ == "__main__":
    # Load parameters
    params = load_params()
    logger.info(params.model_dump())

    # Load model
    model, processor = load_model(
        pretrained_model_name_or_path=params.model.name_or_path,
        cache_dir=params.model.weights_dir,
    )

    if params.model.device is None:
        params.model.device = get_device()

    model.to(params.model.device)

    logger.info(f"Model running on: {model.device}")

    # Test text embedding
    text_embeddings = get_text_embeddings(
        text_input=params.model.test_sentences, model=model, processor=processor
    )
    logger.info(f"Text embeddings shape: {text_embeddings.shape}")

    # Test image embedding
    test_image = Image.open(params.model.test_image)

    image_embeddings = get_image_embeddings(
        image_input=test_image, model=model, processor=processor
    )
    logger.info(f"Image embeddings shape: {image_embeddings.shape}")