import json
from pathlib import Path
from typing import List, Literal, Tuple

from pydantic import BaseModel

PARAMS_PATH = Path("./params.json")


class HealtStatus(BaseModel):
    database: bool
    model: bool
    overall: bool


class EmbeddingResponse(BaseModel):
    type: Literal["text", "image"]
    shape: Tuple
    value: List


class BackendParams(BaseModel):
    model_endpoint_url: str


class EmbeddingEndpoints:
    TEXT: str = "/embedding/text"
    IMAGE: str = "/embedding/image"


class DatabaseEndpoints:
    CREATE_COLLECTION: str = "/create/collection"


class EnpointPaths:
    ROOT: str = "/"
    HEALTH: str = "/health"
    EMBEDDING = EmbeddingEndpoints
    DATABASE = DatabaseEndpoints


def load_params(params_path: Path = PARAMS_PATH) -> BackendParams:
    if not params_path.exists():
        raise FileNotFoundError(f"Backen params file do not exist: {params_path}")
    else:
        with open(file=params_path, mode="r", encoding="utf-8") as file:
            params = json.loads(file.read())

        return BackendParams(**params)


if __name__ == "__main__":
    from loguru import logger

    # Load parameters
    params = load_params(params_path=PARAMS_PATH)
    logger(params.model_dump())
