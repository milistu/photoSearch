import json
from pathlib import Path
from typing import List, Tuple, Literal

import torch
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

PARAMS_PATH = Path("./backend_params.json")


class HealtStatus(BaseModel):
    database: bool
    model: bool
    overall: bool


class EmbeddingResponse(BaseModel):
    type: Literal["text", "image"]
    shape: Tuple
    value: List


class ModelParams(BaseModel):
    name_or_path: str
    weights_dir: str
    test_sentences: List[str]
    test_image: str
    device: str = None


class BackendParams(BaseModel):
    model: ModelParams


def load_model(
    pretrained_model_name_or_path: str, cache_dir: str
) -> Tuple[CLIPModel, CLIPProcessor]:
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


def load_params(params_path: Path = PARAMS_PATH) -> BackendParams:
    if not params_path.exists():
        raise FileNotFoundError(f"Backen params file do not exist: {params_path}")
    else:
        with open(file=params_path, mode="r", encoding="utf-8") as file:
            params = json.loads(file.read())

        return BackendParams(**params)


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


if __name__ == "__main__":
    from pprint import pprint

    # Load parameters
    params = load_params(params_path=PARAMS_PATH)
    pprint(params.model_dump())

    # Load model
    model, processor = load_model(
        pretrained_model_name_or_path=params.model.name_or_path,
        cache_dir=params.model.weights_dir,
    )
    if params.model.device is None:
        params.model.device = get_device()

    model.to(params.model.device)

    pprint(f"Model running on: {model.device}")

    # Test text embedding
    text_embeddings = get_text_embeddings(
        text_input=params.model.test_sentences, model=model, processor=processor
    )
    pprint(f"Text embeddings shape: {text_embeddings.shape}")

    # Test image embedding
    test_image = Image.open(params.model.test_image)

    image_embeddings = get_image_embeddings(
        image_input=[test_image] * 2, model=model, processor=processor
    )
    pprint(f"Image embeddings shape: {image_embeddings.shape}")
