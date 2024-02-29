import json
from pathlib import Path
from typing import List, Literal, Tuple

import torch
from pydantic import BaseModel

PARAMS_PATH = Path("./params.json")


class HealtStatus(BaseModel):
    healthy: bool

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


class Params(BaseModel):
    model: ModelParams


def load_params(params_path: Path = PARAMS_PATH) -> Params:
    if not params_path.exists():
        raise FileNotFoundError(f"Backen params file do not exist: {params_path}")
    else:
        with open(file=params_path, mode="r", encoding="utf-8") as file:
            params = json.loads(file.read())

        return Params(**params)
    
def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device