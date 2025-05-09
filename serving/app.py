import os
import glob
import csv
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
# Directories where CSV embeddings are mounted
USER_CSV_DIR = os.getenv("USER_CSV_DIR", "/home/jovyan/data/user_latent_vectors")
ITEM_CSV_DIR = os.getenv("ITEM_CSV_DIR", "/home/jovyan/data/item_latent_vectors")

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

user_embeddings = _load_user_embeddings()
item_embeddings = _load_item_embeddings()

if not user_embeddings:
    raise RuntimeError(f"No user embeddings loaded from {USER_CSV_DIR}")
if not item_embeddings:
    raise RuntimeError(f"No item embeddings loaded from {ITEM_CSV_DIR}")

rest_ids, rest_embs = zip(*item_embeddings)
rest_embs = np.stack(rest_embs, axis=0)

# ─── FASTAPI SETUP ────────────────────────────────────────────────────────────
app = FastAPI(
    title="In-Memory RecSys API",
    description="Service for recommending restaurants using precomputed embeddings",
    version="1.0.0"
)

class RecommendRequest(BaseModel):
    user_id: int
    k: int = 10

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

    # Commented out actual recommendation building for now
    # recs = [
    #     Recommendation(business_id=rest_ids[i], score=float(scores[i]))
    #     for i in idxs
    # ]
    # return RecommendResponse(user_id=req.user_id, recommendations=recs)

    # Return raw indices for debugging
    return {"user_id": req.user_id, "idxs": idxs.tolist()}
