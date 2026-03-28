from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import numpy as np
import shutil
from typing import List, Optional
from pydantic import BaseModel

from data_model import DataManager
from engine import EmbeddingEngine

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
data_manager = DataManager()
# Set a default image root if it exists locally
default_root = "Salamanders_2025-04/images/SalamanderID2025/database/images"
if os.path.exists(default_root):
    data_manager.image_root = default_root

engine = EmbeddingEngine()
embedding_progress = 0

class UpdateIDRequest(BaseModel):
    new_id: str

@app.post("/api/load-metadata")
async def load_metadata(file: UploadFile = File(...)):
    # Save uploaded file to a temporary location
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        data_manager.load_metadata(temp_path, data_manager.image_root)
        # We can remove the temp file or keep it as the current metadata file
        data_manager.file_path = temp_path
        return {"message": "Metadata loaded successfully", "count": len(data_manager.df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/image-root")
async def set_image_root(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="Path does not exist")
    data_manager.image_root = path
    if data_manager.df is not None:
        data_manager.load_metadata(data_manager.file_path, path)
    return {"message": "Image root updated"}

def generate_embeddings_task():
    global embedding_progress
    def progress_callback(p):
        global embedding_progress
        embedding_progress = p
    
    image_paths = data_manager.df['full_path'].tolist()
    bboxes = data_manager.get_bboxes()
    engine.generate_embeddings(image_paths, bboxes, progress_callback)

@app.post("/api/tasks/generate-embeddings")
async def start_embeddings(background_tasks: BackgroundTasks):
    if data_manager.df is None:
        raise HTTPException(status_code=400, detail="Metadata not loaded")
    background_tasks.add_task(generate_embeddings_task)
    return {"message": "Embedding generation started"}

@app.get("/api/tasks/embeddings-status")
async def get_embeddings_status():
    return {"progress": embedding_progress, "ready": engine.index is not None}

@app.get("/api/identities")
async def get_identities(search: str = ""):
    if data_manager.df is None:
        return []
    
    df = data_manager.df
    unique_ids = df['Identity ID'].unique()
    
    if search:
        unique_ids = [uid for uid in unique_ids if search.lower() in str(uid).lower()]
    
    results = []
    for uid in unique_ids[:200]: # Limit for performance
        subset = df[df['Identity ID'] == uid]
        row = subset.iloc[0]
        results.append({
            "id": str(uid),
            "count": len(subset),
            "representative_image": row['filename'],
            "index": int(subset.index[0])
        })
    return results

@app.get("/api/identities/{identity_id}/images")
async def get_identity_images(identity_id: str):
    if data_manager.df is None:
        return []
    
    # identity_id might be a number or string
    subset = data_manager.df[data_manager.df['Identity ID'].astype(str) == identity_id]
    results = []
    for idx, row in subset.iterrows():
        results.append({
            "index": int(idx),
            "filename": row['filename'],
            "id": str(row['Identity ID'])
        })
    return results

@app.get("/api/images/{index}/validation")
async def get_validation_data(index: int, k: int = 5):
    if data_manager.df is None:
        raise HTTPException(status_code=400, detail="Metadata not loaded")
    if engine.index is None:
        raise HTTPException(status_code=400, detail="Embeddings not generated. Please run 'Embeddings' task first.")
    
    if index >= len(data_manager.df):
        raise HTTPException(status_code=404, detail="Index out of range")
    
    query_row = data_manager.df.iloc[index]
    
    # Find neighbors
    D, I = engine.index.search(engine.embeddings[index:index+1], k + 1)
    
    neighbors = []
    for i in range(len(I[0])):
        n_idx = int(I[0][i])
        if n_idx == index:
            continue
        
        n_row = data_manager.df.iloc[n_idx]
        neighbors.append({
            "index": n_idx,
            "id": str(n_row['Identity ID']),
            "filename": n_row['filename'],
            "score": float(D[0][i])
        })
    
    return {
        "query": {
            "index": index,
            "id": str(query_row['Identity ID']),
            "filename": query_row['filename']
        },
        "neighbors": neighbors[:k]
    }

@app.patch("/api/images/{index}/id")
async def update_image_id(index: int, req: UpdateIDRequest):
    if data_manager.df is None:
        raise HTTPException(status_code=400, detail="Metadata not loaded")
    
    try:
        data_manager.update_id(index, req.new_id)
        return {"message": "ID updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasks/detect-errors")
async def detect_errors():
    if engine.index is None:
        raise HTTPException(status_code=400, detail="Embeddings not generated")
    
    D, I = engine.find_nearest_neighbors(k=5)
    errors = data_manager.detect_errors(D, I)
    
    results = []
    for idx in errors:
        row = data_manager.df.iloc[idx]
        results.append({
            "index": int(idx),
            "filename": row['filename'],
            "id": str(row['Identity ID'])
        })
    return results

@app.post("/api/tasks/split-data")
async def split_data():
    try:
        train_idx, test_idx = data_manager.split_data()
        return {
            "train_count": len(train_idx),
            "test_count": len(test_idx)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/images/serve/{filename}")
async def serve_image(filename: str):
    # Try current directory first, then default root if it exists
    search_dirs = [data_manager.image_root] if data_manager.image_root else []
    
    for base_dir in search_dirs:
        file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, headers={"Cross-Origin-Resource-Policy": "cross-origin"})
    
    raise HTTPException(status_code=404, detail=f"Image {filename} not found")

@app.get("/api/export")
async def export_csv():
    path = "corrected_metadata_web.csv"
    data_manager.export_csv(path)
    return FileResponse(path, filename="corrected_metadata.csv")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
