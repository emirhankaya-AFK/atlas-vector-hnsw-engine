# Atlas Vector — Technical Walkthrough & Verification Report

This document outlines the architecture, algorithmic verification, test execution outputs, quantitative benchmark results, and engineering limitations for the **Atlas Vector** project.

---

## 1. System Architecture & Components

Atlas Vector is built around 4 primary layers:

### A. Core Indexing Engine (`src/atlas_vector/index/`)
- **`hnsw.py`**:
  - Implements Hierarchical Navigable Small World graphs from scratch.
  - Multi-layer graph routing from `max_level` down to layer 0.
  - Configurable parameters: $M$ (default 16), $M_0 = 2M$ (default 32), $m_L = 1/\ln(M)$, $efConstruction$ (default 100), $efSearch$ (default 50), random seed.
  - Algorithm 2 (`SEARCH-LAYER`) using Python's `heapq` with candidate min-heaps and nearest neighbor max-heaps for $O(\log ef)$ element extraction.
  - Algorithm 4 (`SELECT-NEIGHBORS-HEURISTIC`) implementing directional diversity pruning to prevent local node clumping.
  - Soft-delete tombstones preserving graph routing integrity.
- **`flat.py`**:
  - Exact brute-force $k$-NN search providing mathematical ground truth for computing Recall@k.
  - Strict semantic parity with HNSW for metadata attribute filtering.
- **`metrics.py`**:
  - Vectorized and scalar Euclidean ($L_2$) and Cosine distance metrics implemented via NumPy.

### B. Persistence & Crash Recovery (`src/atlas_vector/storage/`)
- **`wal.py`**:
  - Append-only Write-Ahead Log storing JSONL records (`upsert`, `delete`).
  - Strict flush on write and error-tolerant replay handling corrupted lines.
- **`snapshot.py`**:
  - Atomic serialization using temporary file replacement to prevent corrupted snapshots during crashes.
  - Serializes index graph state, levels, links, vector arrays, and metadata dictionaries.
- **`engine.py`**:
  - Coordinates `VectorCollection` instances and multi-collection registry in `AtlasEngine`.
  - Crash recovery protocol: loads binary snapshot if present, then replays trailing WAL entries.

### C. REST API Gateway (`api/main.py`)
- FastAPI endpoints for collection management, batch and single upserts, query execution with metadata filters, vector deletion, statistics, and snapshot triggers.
- Robust HTTP status codes: 201 for creation, 404 for missing resources, 409 for conflicts, and 422 for dimension mismatches.

### D. Interactive Evaluation Lab (`dashboard/app.py`)
- Streamlit application featuring hyperparameter tuning, synthetic dataset generation, live Approximate vs. Exact query comparison, 2D PCA projection, and interactive Pareto frontier visualization.

---

## 2. Test Execution Verification

The test suite consists of **33 unit, integration, and smoke tests** across four test modules:
- `tests/test_metrics.py`: Scalar and vectorized distance metrics, orthogonal/anti-parallel vectors.
- `tests/test_hnsw.py`: Graph creation, seed determinism, neighbor degree bounds, heuristic pruning, tombstones, and filtering.
- `tests/test_storage.py`: WAL append/replay, corruption tolerance, atomic snapshots, crash recovery, and multi-collection engine.
- `tests/test_api.py`: FastAPI endpoints, validation errors, batch ingestion, query filtering, and full lifecycle smoke test.

### Pytest Terminal Output:
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.4.1, pluggy-1.6.0 -- C:\Users\emirh\AppData\Local\Programs\Python\Python311\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\emirh\Desktop\Kodlar\atlas-vector-hnsw-engine
configfile: pyproject.toml
plugins: anyio-4.14.2, asyncio-1.4.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 33 items

tests/test_api.py::test_health_endpoint PASSED                           [  3%]
tests/test_api.py::test_create_collection_and_list PASSED                [  6%]
tests/test_api.py::test_create_duplicate_collection_conflict PASSED      [  9%]
tests/test_api.py::test_upsert_vector_success_and_query PASSED           [ 12%]
tests/test_api.py::test_upsert_invalid_dimension_raises_422 PASSED       [ 15%]
tests/test_api.py::test_upsert_to_non_existent_collection_404 PASSED     [ 18%]
tests/test_api.py::test_batch_upsert_vectors PASSED                      [ 21%]
tests/test_api.py::test_query_with_metadata_filtering PASSED             [ 24%]
tests/test_api.py::test_delete_vector_and_stats PASSED                   [ 27%]
tests/test_api.py::test_trigger_snapshot_endpoint PASSED                 [ 30%]
tests/test_api.py::test_full_api_smoke_lifecycle PASSED                  [ 33%]
tests/test_hnsw.py::test_hnsw_insertion_and_search_l2 PASSED             [ 36%]
tests/test_hnsw.py::test_hnsw_insertion_and_search_cosine PASSED         [ 39%]
tests/test_hnsw.py::test_deterministic_seed PASSED                       [ 42%]
tests/test_hnsw.py::test_high_recall_against_flat_baseline PASSED        [ 45%]
tests/test_hnsw.py::test_neighbor_limits_m_and_m0 PASSED                 [ 48%]
tests/test_hnsw.py::test_heuristic_neighbor_diversity PASSED             [ 51%]
tests/test_hnsw.py::test_soft_delete_tombstone PASSED                    [ 54%]
tests/test_hnsw.py::test_soft_delete_entry_point PASSED                  [ 57%]
tests/test_hnsw.py::test_metadata_filtering PASSED                       [ 60%]
tests/test_hnsw.py::test_invalid_dimension_error PASSED                  [ 63%]
tests/test_hnsw.py::test_hnsw_stats_structure PASSED                     [ 66%]
tests/test_metrics.py::test_pairwise_l2 PASSED                           [ 69%]
tests/test_metrics.py::test_pairwise_cosine PASSED                       [ 72%]
tests/test_metrics.py::test_vectorized_l2_matches_pairwise PASSED        [ 75%]
tests/test_metrics.py::test_vectorized_cosine_matches_pairwise PASSED    [ 78%]
tests/test_storage.py::test_wal_append_and_replay PASSED                 [ 81%]
tests/test_storage.py::test_wal_handles_corrupted_line PASSED            [ 84%]
tests/test_storage.py::test_wal_clear PASSED                             [ 87%]
tests/test_storage.py::test_snapshot_save_and_load PASSED                [ 90%]
tests/test_storage.py::test_snapshot_non_existent_file PASSED            [ 93%]
tests/test_storage.py::test_crash_recovery_snapshot_plus_wal PASSED      [ 96%]
tests/test_storage.py::test_multi_collection_engine PASSED               [100%]

======================== 33 passed in 2.64s ========================
```

### Ruff Code Quality Output:
```
All checks passed!
```

---

## 3. Quantitative Benchmark Output

The benchmark runner (`tests/evaluation/run_benchmark.py`) executed all three sections and validated the algorithmic assertions:

```
================================================================================
  ATLAS VECTOR — HNSW ANN SEARCH ENGINE & BENCHMARK LAB
  Reproducible Quantitative Benchmark Execution
================================================================================
Platform: Windows 10 (AMD64)
Python:   3.11.9
CPU:      8 logical cores
--------------------------------------------------------------------------------

[SECTION 1] Scalability Benchmark Across Dataset Sizes (D=64, Cosine, M=16, efC=100, efS=50):
  --> Benchmarking N=500 vectors ... Done in 11.83s (Recall@10: 1.000, QPS: 455.3)
  --> Benchmarking N=2000 vectors ... Done in 81.28s (Recall@10: 0.979, QPS: 227.9)
  --> Benchmarking N=5000 vectors ... Done in 272.56s (Recall@10: 0.895, QPS: 180.8)

================================================================================
Size (N)   | Build (s)  | Build QPS   | Recall@1   | Recall@10  | QPS      | p50 (ms)  | p95 (ms) 
--------------------------------------------------------------------------------
500        | 11.83      | 42.3        | 1.000      | 1.000      | 455.3    | 2.083     | 3.038    
2000       | 81.28      | 24.6        | 1.000      | 0.979      | 227.9    | 4.373     | 5.133    
5000       | 272.56     | 18.3        | 0.950      | 0.895      | 180.8    | 5.278     | 6.938    
================================================================================

[SECTION 2] Recall vs. QPS Trade-Off Exploration (N=2000, D=64):
efSearch   | Recall@10    | QPS        | p50 Latency (ms)   | p95 Latency (ms)  
---------------------------------------------------------------------------
10         | 0.692        | 625.8      | 1.477              | 2.507             
20         | 0.848        | 443.0      | 2.179              | 3.121             
50         | 0.979        | 241.6      | 3.886              | 5.092             
100        | 0.999        | 158.3      | 5.970              | 7.823             
200        | 1.000        | 131.7      | 7.446              | 9.387             
---------------------------------------------------------------------------

[SECTION 3] Persistence & Durability Verification:
  ✓ Snapshot serialization time : 7.75 ms
  ✓ Snapshot deserialization time: 13.35 ms
  ✓ Restored vectors verified    : 500 vectors

[BENCHMARK ASSERTIONS]
  • Checking Minimum Recall@10 (Measured: 0.895, Required: >= 0.80) ... PASSED!
  • Checking efSearch monotonic Recall increase (ef=10 vs ef=200) ... PASSED!

✓ ALL QUANTITATIVE BENCHMARK CRITERIA MET.
```

---

## 4. Known Limitations & Trade-Offs

1. **CPython Interpreter Bound**: Graph traversals and candidate heap manipulations incur Python bytecode interpreter overhead. A production engine requires C++/Rust with AVX-512 / NEON hardware intrinsics.
2. **Memory Footprint**: Python dictionaries and sets holding neighbor links use substantial pointer overhead compared to flat contiguous memory buffers.
3. **Filter Traversal Saturation**: If a metadata filter matches a very small percentage of items, candidate lists can get saturated by non-matching vectors before reaching top-$k$ matches.
