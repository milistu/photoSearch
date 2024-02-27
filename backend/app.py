import io

from utils import (
    EmbeddingResponse,
    HealtStatus,
    get_device,
    get_image_embeddings,
    get_text_embeddings,
    load_model,
    load_params,
)
from fastapi import FastAPI, File, UploadFile
from PIL import Image

params = load_params()

if params.model.device is None:
    params.model.device = get_device()

model, processor = load_model(
    pretrained_model_name_or_path=params.model.name_or_path,
    cache_dir=params.model.weights_dir,
)
model.to(device=params.model.device)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


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


@app.get("/health", response_model=HealtStatus)
async def health_check() -> HealtStatus:
    db_status = True
    model_status = True if model.name_or_path == params.model.name_or_path else False
    overall_status = db_status and model_status
    return HealtStatus(database=db_status, model=model_status, overall=overall_status)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=9000, reload=True)
