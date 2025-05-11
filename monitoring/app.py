import os
import glob
import csv
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
# Directories where CSV embeddings are mounted
USER_CSV_DIR = os.getenv("USER_CSV_DIR", "/app/data/user_latent_vectors")
ITEM_CSV_DIR = os.getenv("ITEM_CSV_DIR", "/app/data/item_latent_vectors")

# ─── LOAD EMBEDDINGS AT STARTUP ───────────────────────────────────────────────
# Load user embeddings into memory
def _load_user_embeddings():
    embeddings = {}
    pattern = os.path.join(USER_CSV_DIR, "*.csv")
    for fp in glob.glob(pattern):
        with open(fp, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    uid = row['user_id']
                    feats = row['features']
                    vals = feats.strip().split(',')
                    emb = np.array([float(x) for x in vals], dtype=np.float32)
                    embeddings[uid] = emb
                except Exception:
                    continue
    return embeddings

# Load item embeddings into memory
def _load_item_embeddings():
    items = []
    pattern = os.path.join(ITEM_CSV_DIR, "*.csv")
    for fp in glob.glob(pattern):
        with open(fp, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    bid = row['business_id']
                    feats = row['features']
                    vals = feats.strip().split(',')
                    emb = np.array([float(x) for x in vals], dtype=np.float32)
                    items.append((bid, emb))
                except Exception:
                    continue
    return items

# Create dummy data if no embeddings are found
def _create_dummy_data():
    # Ensure directories exist
    os.makedirs(USER_CSV_DIR, exist_ok=True)
    os.makedirs(ITEM_CSV_DIR, exist_ok=True)
    
    # Create a dummy user embedding
    dummy_user_file = os.path.join(USER_CSV_DIR, "dummy_user.csv")
    if not os.path.exists(dummy_user_file):
        with open(dummy_user_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['user_id', 'features'])
            writer.writerow(['user1', '0.1,0.2,0.3,0.4,0.5'])
    
    # Create a dummy item embedding
    dummy_item_file = os.path.join(ITEM_CSV_DIR, "dummy_item.csv")
    if not os.path.exists(dummy_item_file):
        with open(dummy_item_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['business_id', 'features'])
            writer.writerow(['business1', '0.5,0.4,0.3,0.2,0.1'])

# Try to load embeddings, create dummy data if none exist
user_embeddings = _load_user_embeddings()
item_embeddings = _load_item_embeddings()

if not user_embeddings or not item_embeddings:
    print("No embeddings found, creating dummy data")
    _create_dummy_data()
    user_embeddings = _load_user_embeddings()
    item_embeddings = _load_item_embeddings()

if not user_embeddings:
    print(f"WARNING: No user embeddings loaded from {USER_CSV_DIR}")
    # Create a minimal dummy user embedding for testing
    user_embeddings = {"dummy_user": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)}

if not item_embeddings:
    print(f"WARNING: No item embeddings loaded from {ITEM_CSV_DIR}")
    # Create a minimal dummy item embedding for testing
    item_embeddings = [("dummy_item", np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32))]

rest_ids, rest_embs = zip(*item_embeddings)
rest_embs = np.stack(rest_embs, axis=0)

# ─── FASTAPI SETUP ────────────────────────────────────────────────────────────
app = FastAPI(
    title="RecSys API",
    description="Service for recommending restaurants using precomputed embeddings",
    version="1.0.0"
)

class RecommendRequest(BaseModel):
    user_id: str
    k: int = 10

# ─── HEALTH CHECK ENDPOINT ────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "healthy", "user_count": len(user_embeddings), "item_count": len(item_embeddings)}

# ─── ENDPOINT (DEBUG) ──────────────────────────────────────────────────────────
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    # Validate user
    if req.user_id not in user_embeddings:
        raise HTTPException(status_code=404, detail=f"User {req.user_id} not found")
    user_emb = user_embeddings[req.user_id]

    # Compute dot products between user embedding and all item embeddings
    scores = rest_embs.dot(user_emb)
    # Get top-k indices
    idxs = np.argsort(scores)[::-1][:req.k]

    # Debug: print scores and idxs to console
    print("Scores:", scores)
    print("Idxs:", idxs)

    # Return raw indices for debugging
    return {"user_id": req.user_id, "idxs": idxs.tolist()}


Instrumentator().instrument(app).expose(app)
