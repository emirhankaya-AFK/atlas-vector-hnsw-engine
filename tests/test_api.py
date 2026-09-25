"""Integration and smoke tests for FastAPI REST API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app, engine
from src.atlas_vector.engine import AtlasEngine


@pytest.fixture(autouse=True)
def clean_engine():
    """Resets in-memory engine collections between tests."""
    engine.collections.clear()
    yield
    engine.collections.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine"] == "atlas-vector-hnsw"


def test_create_collection_and_list(client: TestClient) -> None:
    payload = {
        "name": "articles",
        "dimension": 4,
        "metric": "cosine",
        "m": 8,
        "ef_construction": 50,
    }
    resp = client.post("/collections", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "articles"
    assert data["dimension"] == 4
    assert data["metric"] == "cosine"

    list_resp = client.get("/collections")
    assert list_resp.status_code == 200
    collections = list_resp.json()["collections"]
    assert len(collections) == 1
    assert collections[0]["name"] == "articles"


def test_create_duplicate_collection_conflict(client: TestClient) -> None:
    payload = {"name": "unique_col", "dimension": 4, "metric": "l2"}
    resp1 = client.post("/collections", json=payload)
    assert resp1.status_code == 201

    resp2 = client.post("/collections", json=payload)
    assert resp2.status_code == 409
    assert "already exists" in resp2.json()["detail"]


def test_upsert_vector_success_and_query(client: TestClient) -> None:
    client.post("/collections", json={"name": "test_search", "dimension": 2, "metric": "l2"})

    # Upsert single vector
    upsert_resp = client.post(
        "/collections/test_search/vectors",
        json={"id": "point_1", "vector": [1.0, 1.0], "metadata": {"tag": "alpha"}},
    )
    assert upsert_resp.status_code == 201
    assert upsert_resp.json()["status"] == "upserted"

    # Query vector
    query_resp = client.post(
        "/collections/test_search/query",
        json={"vector": [0.9, 1.1], "k": 1, "ef_search": 20},
    )
    assert query_resp.status_code == 200
    results = query_resp.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == "point_1"
    assert results[0]["metadata"]["tag"] == "alpha"


def test_upsert_invalid_dimension_raises_422(client: TestClient) -> None:
    client.post("/collections", json={"name": "dim_test", "dimension": 3, "metric": "l2"})

    # Submit 2D vector instead of required 3D
    resp = client.post(
        "/collections/dim_test/vectors",
        json={"id": "bad_vec", "vector": [1.0, 2.0]},
    )
    assert resp.status_code == 422
    assert "dimension" in resp.json()["detail"].lower()


def test_upsert_to_non_existent_collection_404(client: TestClient) -> None:
    resp = client.post(
        "/collections/missing_collection/vectors",
        json={"id": "vec1", "vector": [1.0, 2.0]},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_batch_upsert_vectors(client: TestClient) -> None:
    client.post("/collections", json={"name": "batch_test", "dimension": 2, "metric": "cosine"})

    batch_payload = {
        "vectors": [
            {"id": "b1", "vector": [1.0, 0.0], "metadata": {"group": 1}},
            {"id": "b2", "vector": [0.0, 1.0], "metadata": {"group": 2}},
            {"id": "b3", "vector": [0.707, 0.707], "metadata": {"group": 1}},
        ]
    }
    resp = client.post("/collections/batch_test/vectors/batch", json=batch_payload)
    assert resp.status_code == 201
    assert resp.json()["count"] == 3


def test_query_with_metadata_filtering(client: TestClient) -> None:
    client.post("/collections", json={"name": "filtered_search", "dimension": 2, "metric": "l2"})

    client.post(
        "/collections/filtered_search/vectors",
        json={"id": "tr1", "vector": [1.0, 1.0], "metadata": {"country": "TR"}},
    )
    client.post(
        "/collections/filtered_search/vectors",
        json={"id": "us1", "vector": [1.05, 1.05], "metadata": {"country": "US"}},
    )

    # Query with where country == TR
    query_resp = client.post(
        "/collections/filtered_search/query",
        json={"vector": [1.0, 1.0], "k": 5, "where": {"country": "TR"}},
    )
    assert query_resp.status_code == 200
    results = query_resp.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == "tr1"


def test_delete_vector_and_stats(client: TestClient) -> None:
    client.post("/collections", json={"name": "del_test", "dimension": 2, "metric": "l2"})
    client.post("/collections/del_test/vectors", json={"id": "v1", "vector": [0.0, 0.0]})
    client.post("/collections/del_test/vectors", json={"id": "v2", "vector": [1.0, 1.0]})

    # Delete v1
    del_resp = client.delete("/collections/del_test/vectors/v1")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # Check stats
    stats_resp = client.get("/collections/del_test/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_vectors"] == 2
    assert stats["active_vectors"] == 1
    assert stats["deleted_vectors"] == 1


def test_trigger_snapshot_endpoint(client: TestClient) -> None:
    client.post("/collections", json={"name": "snap_test", "dimension": 2, "metric": "l2"})
    client.post("/collections/snap_test/vectors", json={"id": "s1", "vector": [1.0, 2.0]})

    snap_resp = client.post("/collections/snap_test/snapshot")
    assert snap_resp.status_code == 200
    assert snap_resp.json()["status"] == "snapshot_created"


def test_full_api_smoke_lifecycle(client: TestClient) -> None:
    # 1. Create collection
    r1 = client.post("/collections", json={"name": "smoke_collection", "dimension": 3, "metric": "cosine"})
    assert r1.status_code == 201

    # 2. Batch upsert
    r2 = client.post(
        "/collections/smoke_collection/vectors/batch",
        json={
            "vectors": [
                {"id": "doc_a", "vector": [1.0, 0.0, 0.0], "metadata": {"domain": "tech"}},
                {"id": "doc_b", "vector": [0.0, 1.0, 0.0], "metadata": {"domain": "finance"}},
                {"id": "doc_c", "vector": [0.0, 0.0, 1.0], "metadata": {"domain": "tech"}},
            ]
        },
    )
    assert r2.status_code == 201

    # 3. Query with filter
    r3 = client.post(
        "/collections/smoke_collection/query",
        json={"vector": [0.9, 0.1, 0.0], "k": 2, "where": {"domain": "tech"}},
    )
    assert r3.status_code == 200
    assert len(r3.json()["results"]) == 2
    assert r3.json()["results"][0]["id"] == "doc_a"

    # 4. Delete vector
    r4 = client.delete("/collections/smoke_collection/vectors/doc_a")
    assert r4.status_code == 200

    # 5. Verify stats
    r5 = client.get("/collections/smoke_collection/stats")
    assert r5.status_code == 200
    assert r5.json()["active_vectors"] == 2


def test_api_restart_preserves_collections_and_vectors(tmp_path, monkeypatch) -> None:
    # 1. Point engine to isolated temporary storage
    test_engine = AtlasEngine(data_dir=tmp_path, auto_recover=False)
    monkeypatch.setattr("api.main.engine", test_engine)

    with TestClient(app) as c1:
        c1.post("/collections", json={"name": "kb_restart", "dimension": 2, "metric": "l2"})
        c1.post("/collections/kb_restart/vectors", json={"id": "v1", "vector": [1.0, 1.0], "metadata": {"lang": "en"}})
        c1.post("/collections/kb_restart/vectors", json={"id": "v2", "vector": [2.0, 2.0], "metadata": {"lang": "tr"}})
        c1.post("/collections/kb_restart/snapshot")
        c1.post("/collections/kb_restart/vectors", json={"id": "v3", "vector": [3.0, 3.0], "metadata": {"lang": "tr"}})
        c1.delete("/collections/kb_restart/vectors/v1")

    # 2. Simulate complete application process reboot
    rebooted_engine = AtlasEngine(data_dir=tmp_path, auto_recover=True)
    monkeypatch.setattr("api.main.engine", rebooted_engine)

    with TestClient(app) as c2:
        list_resp = c2.get("/collections")
        assert list_resp.status_code == 200
        cols = [col["name"] for col in list_resp.json()["collections"]]
        assert "kb_restart" in cols

        query_resp = c2.post(
            "/collections/kb_restart/query",
            json={"vector": [2.9, 3.1], "k": 5},
        )
        assert query_resp.status_code == 200
        result_ids = [r["id"] for r in query_resp.json()["results"]]
        assert "v3" in result_ids
        assert "v2" in result_ids
        assert "v1" not in result_ids

        stats_resp = c2.get("/collections/kb_restart/stats")
        assert stats_resp.status_code == 200
        assert stats_resp.json()["active_vectors"] == 2
