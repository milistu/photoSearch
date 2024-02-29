import os

import requests
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from utils import EmbeddingResponse, EnpointPaths, HealtStatus, load_params

MODEL_URL = os.environ["MODEL_URL"]
params = load_params()

app = FastAPI()


@app.get(EnpointPaths.ROOT)
async def root():
    print(f"Model endpoint url: {os.environ['MODEL_URL']}")
    return {"message": "Hello World"}


@app.get(EnpointPaths.HEALTH, response_model=HealtStatus)
async def health_check() -> HealtStatus:
    db_status = True
    response = requests.get(url=MODEL_URL + EnpointPaths.HEALTH)
    if response.status_code == 200:
        model_status = response.json()
        print("Health Check:", model_status)
    else:
        print("Error:", response.text)

    overall_status = db_status and model_status["healthy"]
    return HealtStatus(
        database=db_status, model=model_status["healthy"], overall=overall_status
    )


@app.post(EnpointPaths.EMBEDDING.TEXT)
async def embedding_text(input_text: str) -> EmbeddingResponse:
    data = {"input_text": input_text}
    response = requests.post(url=MODEL_URL + EnpointPaths.EMBEDDING.TEXT, params=data)

    if response.status_code == 200:
        print("Text Embedding:", response.json())
        logger.info("Text Embedding:", response.json())
        return response.json()
    else:
        print("Error:", response.text)
        logger.info("Error:", response.text)
        return response.text


@app.post(EnpointPaths.EMBEDDING.IMAGE)
async def embedding_image(image_file: UploadFile = File(...)) -> EmbeddingResponse:
    image_bytes = await image_file.read()
    # input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    files = {"image_file": image_bytes}

    response = requests.post(url=MODEL_URL + EnpointPaths.EMBEDDING.IMAGE, files=files)

    if response.status_code == 200:
        print("Image Embedding:", response.json())
        logger.info("Image Embedding:", response.json())
        return response.json()
    else:
        print("Error:", response.text)
        logger.info("Error:", response.text)
        return response.text


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=9000, reload=True)
