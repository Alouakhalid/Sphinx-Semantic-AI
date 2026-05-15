from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os

from model_utils import predict_similarity, train_on_pair, get_model_stats, get_training_logs

app = FastAPI(
    title="Sphinx Semantic AI",
    description="Semantic Intelligence, Reimagined",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

class PredictRequest(BaseModel):
    sentence1: str
    sentence2: str

class TrainRequest(BaseModel):
    sentence1: str
    sentence2: str
    label: float

@app.get("/")
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Sphinx Semantic AI API is running. index.html not found in /static/"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Sphinx Semantic AI", "version": "1.0.0"}

@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.sentence1 or not req.sentence2:
        raise HTTPException(status_code=400, detail="Both sentences are required.")
    
    try:
        score = predict_similarity(req.sentence1, req.sentence2)
        return {
            "similarity": round(score, 6),
            "percentage": round(score * 100, 2),
            "sentence1": req.sentence1,
            "sentence2": req.sentence2,
            "interpretation": interpret_score(score)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train(req: TrainRequest, background_tasks: BackgroundTasks):
    if not req.sentence1 or not req.sentence2:
        raise HTTPException(status_code=400, detail="Both sentences are required.")
    if not (0.0 <= req.label <= 1.0):
        raise HTTPException(status_code=400, detail="Label must be between 0.0 and 1.0")

    background_tasks.add_task(train_on_pair, req.sentence1, req.sentence2, req.label)
    
    return {
        "status": "success",
        "message": "Training started in background",
        "sentence1": req.sentence1,
        "sentence2": req.sentence2,
        "label": req.label
    }

@app.get("/stats")
async def stats():
    try:
        return get_model_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
async def logs():
    try:
        return {"logs": get_training_logs()[-50:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def interpret_score(score: float) -> str:
    if score >= 0.9:
        return "Nearly identical meaning"
    elif score >= 0.75:
        return "Very similar meaning"
    elif score >= 0.6:
        return "Moderately similar"
    elif score >= 0.4:
        return "Somewhat related"
    elif score >= 0.2:
        return "Weakly related"
    else:
        return "Unrelated or opposite meaning"

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=False)
