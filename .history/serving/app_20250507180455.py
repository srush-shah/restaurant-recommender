from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
import numpy as np
from sqlalchemy import create_engine, text

# ─── FASTAPI APP ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Transformer‑Based RecSys API",
    description="Given user_id & city, returns scored recommendations via a PyTorch transformer model",
    version="1.0.0",
)

# ─── REQUEST/RESPONSE SCHEMAS ─────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    user_id: int
    city: str
    k: int = 10

class Recommendation(BaseModel):
    business_id: str
    score: float

class RecommendResponse(BaseModel):
    user_id: int
    city: str
    recommendations: list[Recommendation]

# ─── DEVICE & MODEL LOADING ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.getenv("MODEL_PATH", "transformer_recommender.pth")
# if you saved the full model:
model = torch.load(MODEL_PATH, map_location=device)  
# OR, if you saved only state_dict:
# from your_model_definition import TransformerRecommender
# model = TransformerRecommender(**your_init_args)
# state = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(state)
model.to(device)
model.eval()

# ─── DATABASE SETUP ───────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/recsys")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def get_user_embedding(user_id: int) -> np.ndarray:
    q = text("SELECT embedding FROM user_embeddings WHERE user_id = :uid")
    with engine.connect() as conn:
        row = conn.execute(q, {"uid": user_id}).fetchone()
    if not row:
        raise HTTPException(404, f"User {user_id} not found")
    return np.frombuffer(row.embedding, dtype=np.float32)

def get_restaurant_embeddings(city: str):
    q = text("SELECT business_id, embedding FROM restaurant_embeddings WHERE city = :c")
    with engine.connect() as conn:
        rows = conn.execute(q, {"c": city}).fetchall()
    if not rows:
        raise HTTPException(404, f"No restaurants in '{city}'")
    ids, embs = zip(*rows)
    return list(ids), np.stack([np.frombuffer(b, dtype=np.float32) for b in embs])

def top_n_by_dot(user_emb, rest_ids, rest_embs, n):
    dots = rest_embs.dot(user_emb)
    idxs = np.argsort(dots)[::-1][:n]
    return [rest_ids[i] for i in idxs]

def score_with_transformer(user_id: int, candidate_ids: list[str]):
    """
    Build whatever input your transformer needs—e.g. user & business embeddings,
    maybe concatenated or as a dict—and then call model.forward().
    """
    # example: assume model takes two tensors (batch_size, D)
    user_emb = get_user_embedding(user_id)
    rest_ids, rest_embs = get_restaurant_embeddings(req.city)  # already done upstream
    # gather only the embeddings for candidate_ids
    # ... then:
    user_batch = torch.tensor([user_emb for _ in candidate_ids], device=device)  # (top_n, D)
    item_batch = torch.tensor([rest_embs[rest_ids.index(b)] for b in candidate_ids], device=device)
    with torch.no_grad():
        # assume model returns raw scores of shape (batch_size,)
        outputs = model(user_batch, item_batch)
        scores = outputs.cpu().numpy().tolist()
    return list(zip(candidate_ids, scores))

# ─── ENDPOINT ─────────────────────────────────────────────────────────────────
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    # 1) Fetch embeddings & pre‑rank
    user_emb   = get_user_embedding(req.user_id)
    rest_ids, rest_embs = get_restaurant_embeddings(req.city)
    top_candidates = top_n_by_dot(user_emb, rest_ids, rest_embs, 100)

    # 2) Re‑score with transformer
    scored = score_with_transformer(req.user_id, top_candidates)

    # 3) Final sort and slice
    scored.sort(key=lambda x: x[1], reverse=True)
    final = scored[:req.k]

    return RecommendResponse(
        user_id=req.user_id,
        city=req.city,
        recommendations=[
            Recommendation(business_id=b, score=float(s)) for b, s in final
        ]
    )