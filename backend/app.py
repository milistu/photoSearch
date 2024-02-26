from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI()

class HealtStatus(BaseModel):
    database: bool
    model: bool
    overall: bool

@app.get("/")
async def root():
    return {"message" : "Hello World"}

@app.get("/health", response_model=HealtStatus)
async def health_check() -> HealtStatus:
    db_status = True
    model_status = True
    overall_status = db_status and model_status
    return HealtStatus(database=db_status, model=model_status, overall=overall_status)