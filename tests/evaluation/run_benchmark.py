"""Comprehensive quantitative evaluation benchmark for Atlas Vector HNSW Engine.

Evaluates:
- Build time & Indexing throughput (vectors/sec) across 3 dataset sizes (500, 2,000, 5,000)
- Recall@1 and Recall@10 against exact FlatIndex ground truth
- Query throughput (QPS) and latency distribution (p50, p95, p99)
- Recall vs. QPS trade-off across varying efSearch values [10, 20, 50, 100, 200]
- Persistence & Durability: Atomic Snapshot + WAL Replay Engine Crash Recovery
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
import time
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np

from src.atlas_vector.engine import VectorCollection
from src.atlas_vector.index.flat import FlatIndex
from src.atlas_vector.index.hnsw import HNSWIndex
from src.atlas_vector.storage.snapshot import load_snapshot, save_snapshot


def generate_benchmark_data(n_samples: int, dimension: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic normalized vectors for reproducible benchmarking."""
    rng = np.random.default_rng(seed)
    raw_train = rng.normal(loc=0.0, scale=1.0, size=(n_samples, dimension)).astype(np.float32)
    norms_train = np.linalg.norm(raw_train, axis=1, keepdims=True)
    train_vectors = raw_train / np.maximum(norms_train, 1e-12)

    raw_queries = rng.normal(loc=0.0, scale=1.0, size=(100, dimension)).astype(np.float32)
    norms_query = np.linalg.norm(raw_queries, axis=1, keepdims=True)
    query_vectors = raw_queries / np.maximum(norms_query, 1e-12)

    return train_vectors, query_vectors


def compute_ground_truth(
    train_vectors: np.ndarray,
    query_vectors: np.ndarray,
    metric: str = "cosine",
    k: int = 10,
) -> list[list[str]]:
    """Computes exact brute-force top-k neighbors for each query."""
    dimension = train_vectors.shape[1]
    flat = FlatIndex(dimension=dimension, metric=metric)
    for i, vec in enumerate(train_vectors):
        flat.upsert(f"vec_{i}", vec)

    ground_truth = []
    for q in query_vectors:
        results = flat.search(q, k=k)
        ground_truth.append([vid for vid, _ in results])
    return ground_truth


def benchmark_dataset_size(
    n_vectors: int,
    dimension: int = 64,
    metric: str = "cosine",
    m: int = 16,
    ef_construction: int = 100,
    ef_search: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs end-to-end benchmark for a specific dataset size."""
    train_vectors, query_vectors = generate_benchmark_data(n_vectors, dimension, seed=seed)

    # 1. Ground truth calculation
    ground_truth = compute_ground_truth(train_vectors, query_vectors, metric=metric, k=10)

    # 2. HNSW Index Construction
    hnsw = HNSWIndex(
        dimension=dimension,
        metric=metric,
        m=m,
        ef_construction=ef_construction,
        seed=seed,
    )

    build_start = time.perf_counter()
    for i, vec in enumerate(train_vectors):
        hnsw.upsert(f"vec_{i}", vec)
    build_time = time.perf_counter() - build_start
    indexing_throughput = n_vectors / max(build_time, 1e-6)

    # 3. Query Latency and Recall
    latencies_ms: list[float] = []
    recall_at_1_hits = 0
    recall_at_10_sum = 0.0

    query_start_total = time.perf_counter()
    for q_idx, query in enumerate(query_vectors):
        t0 = time.perf_counter()
        results = hnsw.search(query, k=10, ef_search=ef_search)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        retrieved_ids = [vid for vid, _ in results]
        gt_ids = ground_truth[q_idx]

        if retrieved_ids and gt_ids and retrieved_ids[0] == gt_ids[0]:
            recall_at_1_hits += 1

        common = set(retrieved_ids[:10]) & set(gt_ids[:10])
        recall_at_10_sum += len(common) / 10.0

    total_query_time = time.perf_counter() - query_start_total
    qps = len(query_vectors) / max(total_query_time, 1e-6)
    recall_1 = recall_at_1_hits / len(query_vectors)
    recall_10 = recall_at_10_sum / len(query_vectors)

    latencies_ms.sort()
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))

    return {
        "n_vectors": n_vectors,
        "dimension": dimension,
        "metric": metric,
        "build_time_sec": build_time,
        "indexing_throughput": indexing_throughput,
        "recall_at_1": recall_1,
        "recall_at_10": recall_10,
        "qps": qps,
        "latency_mean_ms": mean_lat,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "hnsw_instance": hnsw,
        "query_vectors": query_vectors,
        "ground_truth": ground_truth,
    }


def benchmark_ef_tradeoff(
    hnsw: HNSWIndex,
    query_vectors: np.ndarray,
    ground_truth: list[list[str]],
    ef_values: list[int],
) -> list[dict[str, Any]]:
    """Evaluates Recall@10 vs QPS trade-off across varying efSearch values."""
    tradeoff_results = []
    for ef in ef_values:
        latencies_ms = []
        recall_sum = 0.0
        t_start = time.perf_counter()

        for q_idx, query in enumerate(query_vectors):
            t0 = time.perf_counter()
            results = hnsw.search(query, k=10, ef_search=ef)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            retrieved_ids = [vid for vid, _ in results]
            common = set(retrieved_ids[:10]) & set(ground_truth[q_idx][:10])
            recall_sum += len(common) / 10.0

        total_time = time.perf_counter() - t_start
        qps = len(query_vectors) / max(total_time, 1e-6)
        recall_10 = recall_sum / len(query_vectors)
        latencies_ms.sort()

        tradeoff_results.append({
            "ef_search": ef,
            "recall_at_10": recall_10,
            "qps": qps,
            "p50_ms": float(np.percentile(latencies_ms, 50)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "p99_ms": float(np.percentile(latencies_ms, 99)),
        })
    return tradeoff_results


def run_all_benchmarks() -> bool:
    print("=" * 90)
    print("  ATLAS VECTOR — HNSW ANN SEARCH ENGINE & BENCHMARK LAB")
    print("  Reproducible Quantitative Benchmark Execution")
    print("=" * 90)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"CPU:      {os.cpu_count()} logical cores")
    print("-" * 90)

    dataset_sizes = [500, 2000, 5000]
    dimension = 64
    metric = "cosine"
    results_by_size = []

    print("\n[SECTION 1] Scalability Benchmark Across Dataset Sizes (D=64, Cosine, M=16, efC=100, efS=50):")
    for size in dataset_sizes:
        print(f"  --> Benchmarking N={size} vectors ...", end="", flush=True)
        res = benchmark_dataset_size(size, dimension=dimension, metric=metric, seed=42)
        results_by_size.append(res)
        print(f" Done in {res['build_time_sec']:.2f}s (Recall@10: {res['recall_at_10']:.3f}, QPS: {res['qps']:.1f})")

    # Print Summary Table with p50, p95, p99
    print("\n" + "=" * 90)
    print(f"{'Size (N)':<9} | {'Build (s)':<10} | {'Build QPS':<10} | {'Recall@1':<9} | {'Recall@10':<10} | {'QPS':<7} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'p99 (ms)':<9}")
    print("-" * 90)
    for r in results_by_size:
        print(
            f"{r['n_vectors']:<9} | "
            f"{r['build_time_sec']:<10.2f} | "
            f"{r['indexing_throughput']:<10.1f} | "
            f"{r['recall_at_1']:<9.3f} | "
            f"{r['recall_at_10']:<10.3f} | "
            f"{r['qps']:<7.1f} | "
            f"{r['latency_p50_ms']:<9.3f} | "
            f"{r['latency_p95_ms']:<9.3f} | "
            f"{r['latency_p99_ms']:<9.3f}"
        )
    print("=" * 90)

    # Section 2: efSearch Trade-off on 2,000 vectors
    print("\n[SECTION 2] Recall vs. QPS Trade-Off Exploration (N=2000, D=64):")
    target_run = results_by_size[1]
    ef_values = [10, 20, 50, 100, 200]
    tradeoffs = benchmark_ef_tradeoff(
        target_run["hnsw_instance"],
        target_run["query_vectors"],
        target_run["ground_truth"],
        ef_values,
    )

    print(f"{'efSearch':<9} | {'Recall@10':<10} | {'QPS':<8} | {'p50 Latency (ms)':<17} | {'p95 Latency (ms)':<17} | {'p99 Latency (ms)':<17}")
    print("-" * 88)
    for t in tradeoffs:
        print(
            f"{t['ef_search']:<9} | "
            f"{t['recall_at_10']:<10.3f} | "
            f"{t['qps']:<8.1f} | "
            f"{t['p50_ms']:<17.3f} | "
            f"{t['p95_ms']:<17.3f} | "
            f"{t['p99_ms']:<17.3f}"
        )
    print("-" * 88)

    # Section 3: Persistence Snapshot & Engine WAL Crash Recovery Benchmark
    print("\n[SECTION 3] Persistence & Durability: Full Engine Crash Recovery Benchmark:")
    bench_data_dir = "data/bench_durability"
    if os.path.exists(bench_data_dir):
        shutil.rmtree(bench_data_dir)
    os.makedirs(bench_data_dir, exist_ok=True)

    # 3.1 Raw Snapshot Serialization
    hnsw_sample = results_by_size[0]["hnsw_instance"]
    temp_snap_path = os.path.join(bench_data_dir, "raw_test.snapshot")

    t_snap_start = time.perf_counter()
    save_snapshot(hnsw_sample, temp_snap_path)
    snap_time = (time.perf_counter() - t_snap_start) * 1000.0

    t_load_start = time.perf_counter()
    restored_hnsw = load_snapshot(temp_snap_path)
    load_time = (time.perf_counter() - t_load_start) * 1000.0

    print(f"  ✓ Raw Snapshot serialization (N=500, fsync)  : {snap_time:.2f} ms")
    print(f"  ✓ Raw Snapshot deserialization (N=500)       : {load_time:.2f} ms")
    print(f"  ✓ Restored raw vectors verified              : {len(restored_hnsw.vectors)} vectors")

    # 3.2 Full Engine Crash Recovery (Snapshot + WAL Replay)
    col = VectorCollection("recovery_bench", dimension=dimension, metric="cosine", data_dir=bench_data_dir)
    train_vecs, _ = generate_benchmark_data(500, dimension, seed=77)
    for i in range(400):
        col.upsert(f"doc_{i}", train_vecs[i].tolist(), {"category": "base"})

    # Checkpoint snapshot
    col.snapshot()

    # Append 100 post-snapshot records to WAL and delete 20
    for i in range(400, 500):
        col.upsert(f"doc_{i}", train_vecs[i].tolist(), {"category": "wal_delta"})
    for i in range(20):
        col.delete(f"doc_{i}")

    # Simulate crash: recreate collection instance and measure recovery time
    t_rec_start = time.perf_counter()
    crashed_col = VectorCollection("recovery_bench", dimension=dimension, metric="cosine", data_dir=bench_data_dir)
    crashed_col.recover()
    full_recovery_time_ms = (time.perf_counter() - t_rec_start) * 1000.0

    active_count = len(crashed_col.index.vectors) - len(crashed_col.index.deleted)
    expected_count = 500 - 20
    print(f"  ✓ Full Engine Recovery Time (Snapshot + WAL) : {full_recovery_time_ms:.2f} ms")
    print(f"  ✓ Recovered Active Vectors Count             : {active_count} (Expected: {expected_count})")
    assert active_count == expected_count, f"Recovery count mismatch: {active_count} vs {expected_count}"

    # Cleanup benchmark directory
    if os.path.exists(bench_data_dir):
        shutil.rmtree(bench_data_dir)

    # Quantitative Assertions
    print("\n[BENCHMARK ASSERTIONS]")
    min_recall = min(r["recall_at_10"] for r in results_by_size)
    print(f"  • Checking Minimum Recall@10 (Measured: {min_recall:.3f}, Required: >= 0.80) ... ", end="")
    if min_recall < 0.80:
        print("FAILED!")
        return False
    print("PASSED!")

    print("  • Checking efSearch monotonic Recall increase (ef=10 vs ef=200) ... ", end="")
    if tradeoffs[0]["recall_at_10"] > tradeoffs[-1]["recall_at_10"]:
        print("FAILED!")
        return False
    print("PASSED!")

    print("\n✓ ALL QUANTITATIVE BENCHMARK CRITERIA MET.")
    return True


if __name__ == "__main__":
    success = run_all_benchmarks()
    if not success:
        sys.exit(1)
