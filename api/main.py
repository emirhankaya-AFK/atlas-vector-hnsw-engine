"""FastAPI REST Service for Atlas Vector HNSW Engine."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Path, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.atlas_vector.engine import AtlasEngine

DATA_DIR = os.environ.get("ATLAS_DATA_DIR", "data")
engine = AtlasEngine(data_dir=DATA_DIR, auto_recover=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensures collection catalog is loaded and indexes recovered on application startup."""
    engine.load_catalog_and_recover()
    yield


app = FastAPI(
    title="Atlas Vector — Educational HNSW ANN Search Engine",
    version="1.0.0",
    description=(
        "Educational and reproducible HNSW approximate nearest neighbor search engine "
        "with Write-Ahead Log (WAL) persistence, exact FlatIndex baseline, and metadata filtering."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CollectionRequest(BaseModel):
    name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]{1,64}$", description="Alphanumeric collection name")
    dimension: int = Field(..., ge=2, le=4096, description="Vector dimension size")
    metric: str = Field(default="cosine", pattern="^(cosine|l2)$", description="Distance metric ('cosine' or 'l2')")
    m: int = Field(default=16, ge=2, le=64, description="Max bi-directional links per node for higher layers")
    ef_construction: int = Field(default=100, ge=4, le=1000, description="Size of dynamic candidate list during build")
    seed: int = Field(default=42, description="Random seed for deterministic level assignment")


class UpsertItem(BaseModel):
    id: str = Field(..., min_length=1, max_length=128, description="Unique vector ID")
    vector: list[float] = Field(..., description="Vector components")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary JSON metadata attributes")


class BatchUpsertRequest(BaseModel):
    vectors: list[UpsertItem] = Field(..., min_length=1, max_length=1000, description="Batch list of vectors")


class QueryRequest(BaseModel):
    vector: list[float] = Field(..., description="Query vector components")
    k: int = Field(default=10, ge=1, le=500, description="Number of nearest neighbors to return")
    ef_search: int = Field(default=50, ge=1, le=2000, description="Dynamic candidate list size during query")
    where: dict[str, Any] | None = Field(default=None, description="Metadata attribute filter criteria")


@app.get("/health", tags=["System"])
def health_check() -> dict[str, Any]:
    """Liveness probe and system summary."""
    return {
        "status": "healthy",
        "engine": "atlas-vector-hnsw",
        "version": "1.0.0",
        "collections_count": len(engine.collections),
    }


@app.get("/collections", tags=["Collections"])
def list_collections() -> dict[str, Any]:
    """Lists all registered vector collections."""
    return {"collections": engine.list_collections()}


@app.post("/collections", status_code=status.HTTP_201_CREATED, tags=["Collections"])
def create_collection(payload: CollectionRequest) -> dict[str, Any]:
    """Creates a new isolated vector collection."""
    try:
        col = engine.create_collection(
            name=payload.name,
            dimension=payload.dimension,
            metric=payload.metric,
            m=payload.m,
            ef_construction=payload.ef_construction,
            seed=payload.seed,
        )
        return {
            "status": "created",
            "name": col.name,
            "dimension": col.dimension,
            "metric": col.metric,
            "m": col.index.m,
            "ef_construction": col.index.ef_construction,
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.post("/collections/{name}/vectors", status_code=status.HTTP_201_CREATED, tags=["Vectors"])
def upsert_vector(
    name: str = Path(..., description="Collection name"),
    payload: UpsertItem = ...,
) -> dict[str, Any]:
    """Upserts a single vector with metadata into the collection."""
    try:
        col = engine.get(name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection '{name}' not found") from None

    try:
        col.upsert(payload.id, payload.vector, payload.metadata)
        return {"status": "upserted", "id": payload.id, "collection": name}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@app.post("/collections/{name}/vectors/batch", status_code=status.HTTP_201_CREATED, tags=["Vectors"])
def batch_upsert_vectors(
    name: str = Path(..., description="Collection name"),
    payload: BatchUpsertRequest = ...,
) -> dict[str, Any]:
    """Batch upserts multiple vectors into the collection."""
    try:
        col = engine.get(name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection '{name}' not found") from None

    upserted_ids = []
    for item in payload.vectors:
        try:
            col.upsert(item.id, item.vector, item.metadata)
            upserted_ids.append(item.id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Error inserting vector '{item.id}': {exc}",
            ) from exc

    return {"status": "batch_upserted", "count": len(upserted_ids), "ids": upserted_ids}


@app.post("/collections/{name}/query", tags=["Vectors"])
def query_vectors(
    name: str = Path(..., description="Collection name"),
    payload: QueryRequest = ...,
) -> dict[str, Any]:
    """Executes an approximate k-NN query with optional metadata filtering."""
    try:
        col = engine.get(name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection '{name}' not found") from None

    try:
        results = col.query(
            vector=payload.vector,
            k=payload.k,
            ef_search=payload.ef_search,
            where=payload.where,
        )
        return {
            "collection": name,
            "count": len(results),
            "k": payload.k,
            "ef_search": payload.ef_search,
            "results": results,
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@app.delete("/collections/{name}/vectors/{vector_id}", tags=["Vectors"])
def delete_vector(
    name: str = Path(..., description="Collection name"),
    vector_id: str = Path(..., description="Vector ID to delete"),
) -> dict[str, Any]:
    """Tombstones a vector in the collection."""
    try:
        col = engine.get(name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection '{name}' not found") from None

    deleted = col.delete(vector_id)
    return {"deleted": deleted, "id": vector_id, "collection": name}


@app.get("/collections/{name}/stats", tags=["Collections"])
def collection_stats(name: str = Path(..., description="Collection name")) -> dict[str, Any]:
    """Retrieves detailed operational and graph structural statistics for a collection."""
    try:
        col = engine.get(name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection '{name}' not found") from None

    return col.stats()


@app.post("/collections/{name}/snapshot", tags=["Persistence"])
def trigger_snapshot(name: str = Path(..., description="Collection name")) -> dict[str, Any]:
    """Forces an immediate index snapshot and truncates the WAL."""
    try:
        col = engine.get(name)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Collection '{name}' not found") from None

    snapshot_path = col.snapshot()
    return {"status": "snapshot_created", "path": snapshot_path, "collection": name}
