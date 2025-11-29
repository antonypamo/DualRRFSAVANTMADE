"""Lightweight FastAPI server for DualRRFSAVANTMADE embeddings.

This module exposes two endpoints:
- /embed: returns embeddings for a list of sentences.
- /similarity: returns cosine similarity scores between a source sentence and targets.

Run with:
    uvicorn deploy:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util


def _resolve_model_path() -> Path:
    env_path = os.getenv("MODEL_PATH")
    return Path(env_path) if env_path else Path(__file__).resolve().parent


model_path = _resolve_model_path()
model = SentenceTransformer(model_path)

app = FastAPI(title="DualRRFSAVANTMADE", version="1.0.0")


class EmbedRequest(BaseModel):
    sentences: List[str] = Field(..., description="Sentences to embed.")
    normalize: bool = Field(True, description="Whether to return normalized embeddings.")


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embeddings: List[List[float]]


@app.get("/")
def read_root() -> dict:
    return {
        "model": str(model_path),
        "dimension": int(model.get_sentence_embedding_dimension()),
        "max_length": int(model.get_max_seq_length()),
        "description": "FastAPI wrapper around the DualRRFSAVANTMADE sentence-transformer model.",
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    embeddings = model.encode(
        request.sentences,
        convert_to_tensor=True,
        normalize_embeddings=request.normalize,
    )
    return EmbedResponse(
        model=str(model_path),
        dimension=int(embeddings.size(-1)),
        embeddings=embeddings.cpu().tolist(),
    )


class SimilarityRequest(BaseModel):
    source: str = Field(..., description="Reference sentence.")
    targets: List[str] = Field(..., description="Sentences to compare against the source.")
    normalize: bool = Field(True, description="Whether to normalize embeddings before computing similarity.")


class SimilarityResponse(BaseModel):
    model: str
    scores: List[float]


@app.post("/similarity", response_model=SimilarityResponse)
def similarity(request: SimilarityRequest) -> SimilarityResponse:
    embeddings = model.encode(
        [request.source, *request.targets],
        convert_to_tensor=True,
        normalize_embeddings=request.normalize,
    )
    source_embedding, target_embeddings = embeddings[0], embeddings[1:]
    scores = util.cos_sim(source_embedding, target_embeddings).squeeze(0)
    return SimilarityResponse(model=str(model_path), scores=scores.tolist())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("deploy:app", host="0.0.0.0", port=8000, reload=False)
