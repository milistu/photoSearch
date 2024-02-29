import io

from clip import get_image_embeddings, get_text_embeddings, load_model
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from utils import (
    EmbeddingResponse,
    HealtStatus,
    get_device,
    load_params,
)

params = load_params()

if params.model.device is None:
    params.model.device = get_device()

model, processor = None, None

model, processor = load_model(
    pretrained_model_name_or_path=params.model.name_or_path,
    cache_dir=params.model.weights_dir,
)
model.to(device=params.model.device)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health", response_model=HealtStatus)
async def health_check() -> HealtStatus:
    health_status = model is not None and processor is not None
    return HealtStatus(healthy=health_status)


@app.post("/embedding/text")
async def embedding_text(input_text: str) -> EmbeddingResponse:
    results = get_text_embeddings(
        text_input=input_text, model=model, processor=processor
    )

    return EmbeddingResponse(type="text", shape=results.shape, value=results.tolist())


@app.post("/embedding/image")
async def embedding_image(image_file: UploadFile = File(...)) -> EmbeddingResponse:
    image_bytes = await image_file.read()
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = get_image_embeddings(
        image_input=input_image, model=model, processor=processor
    )

    return EmbeddingResponse(type="image", shape=results.shape, value=results.tolist())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=9000, reload=True)
