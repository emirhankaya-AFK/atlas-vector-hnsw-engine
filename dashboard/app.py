"""Atlas Vector — Educational & Reproducible HNSW ANN Search Engine Dashboard.

Interactive Laboratory:
- Hyperparameter Tuning (M, efConstruction, efSearch)
- Synthetic Dataset Generator (Clustered, Gaussian, Uniform)
- Live Approximate vs. Exact Ground Truth Query Comparison
- 2D PCA Dimensionality Reduction Projection
- Interactive Recall vs. QPS Trade-Off Curves
- Dark SCADA / Terminal Aesthetic
"""
from __future__ import annotations

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.atlas_vector.index.flat import FlatIndex
from src.atlas_vector.index.hnsw import HNSWIndex


def compute_pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Computes 2D PCA projection using pure NumPy SVD without external sklearn dependency."""
    centered = matrix - np.mean(matrix, axis=0)
    # Singular Value Decomposition
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    # Project onto top 2 principal components
    return centered @ vt[:2].T


# Streamlit Page Config
st.set_page_config(
    page_title="Atlas Vector — HNSW Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Terminal Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }
    .stMetric {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px;
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-size: 0.85rem;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-family: monospace;
    }
    .badge-exact {
        background-color: #064e3b;
        color: #34d399;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-hnsw {
        background-color: #0c4a6e;
        color: #38bdf8;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ Atlas Vector — HNSW ANN & Benchmark Lab")
st.caption(
    "Educational & Reproducible Hierarchical Navigable Small World (HNSW) Exploration Lab. "
    "Compare approximate graph search against brute-force exact baseline."
)

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Index & Data Parameters")
    n_vectors = st.slider("Dataset Size (N)", min_value=100, max_value=2000, value=500, step=100)
    dimension = st.selectbox("Dimension (D)", options=[8, 16, 32, 64], index=1)
    metric = st.selectbox("Distance Metric", options=["cosine", "l2"], index=0)
    distribution = st.selectbox("Dataset Geometry", options=["Gaussian Blobs (Clusters)", "Standard Normal", "Uniform"], index=0)

    st.subheader("HNSW Graph Hyperparameters")
    param_m = st.slider("M (Max Links / Node)", min_value=4, max_value=32, value=12, step=2)
    param_ef_c = st.slider("efConstruction", min_value=16, max_value=200, value=64, step=8)
    param_ef_s = st.slider("efSearch (Query Depth)", min_value=10, max_value=200, value=40, step=10)
    seed = st.number_input("Random Seed", value=42, step=1)

    rebuild_btn = st.button("🔄 Rebuild Dataset & Graph", use_container_width=True)

# Cache / Session State for Dataset & Index
if "vectors" not in st.session_state or rebuild_btn or st.session_state.get("last_n") != n_vectors:
    st.session_state["last_n"] = n_vectors
    rng = np.random.default_rng(seed)

    if distribution == "Gaussian Blobs (Clusters)":
        n_clusters = 5
        centers = rng.normal(size=(n_clusters, dimension)) * 2.0
        data = []
        for i in range(n_vectors):
            cluster_id = i % n_clusters
            vec = centers[cluster_id] + rng.normal(scale=0.3, size=dimension)
            data.append(vec)
        raw_matrix = np.array(data, dtype=np.float32)
    elif distribution == "Standard Normal":
        raw_matrix = rng.normal(size=(n_vectors, dimension)).astype(np.float32)
    else:
        raw_matrix = rng.uniform(low=-1.0, high=1.0, size=(n_vectors, dimension)).astype(np.float32)

    if metric == "cosine":
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        raw_matrix = raw_matrix / np.maximum(norms, 1e-12)

    st.session_state["vectors"] = raw_matrix

    # Build Exact Flat Index
    flat_idx = FlatIndex(dimension=dimension, metric=metric)
    for i, vec in enumerate(raw_matrix):
        category = "technical" if i % 2 == 0 else "general"
        flat_idx.upsert(f"node_{i}", vec, metadata={"category": category, "index": i})
    st.session_state["flat_index"] = flat_idx

    # Build HNSW Index
    hnsw_idx = HNSWIndex(
        dimension=dimension,
        metric=metric,
        m=param_m,
        ef_construction=param_ef_c,
        seed=seed,
    )
    t_start = time.perf_counter()
    for i, vec in enumerate(raw_matrix):
        category = "technical" if i % 2 == 0 else "general"
        hnsw_idx.upsert(f"node_{i}", vec, metadata={"category": category, "index": i})
    build_time = time.perf_counter() - t_start

    st.session_state["hnsw_index"] = hnsw_idx
    st.session_state["build_time"] = build_time

    # Pre-compute 2D PCA coordinates
    st.session_state["pca_coords"] = compute_pca_2d(raw_matrix)

vectors = st.session_state["vectors"]
flat_idx: FlatIndex = st.session_state["flat_index"]
hnsw_idx: HNSWIndex = st.session_state["hnsw_index"]
pca_coords: np.ndarray = st.session_state["pca_coords"]
build_time = st.session_state["build_time"]

# Top Overview Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Indexed Vectors", f"{len(vectors):,}")
with c2:
    st.metric("Index Build Time", f"{build_time:.3f} s", f"{len(vectors)/max(build_time, 1e-6):.0f} vec/s")
with c3:
    st.metric("HNSW Max Level", f"L{hnsw_idx.max_level}")
with c4:
    st.metric("Graph Direct Edges", f"{hnsw_idx.stats()['total_directed_edges']:,}")

# Tab Layout
tab1, tab2, tab3 = st.tabs(["🎯 2D PCA Space & Live Query", "📈 Recall vs. QPS Trade-Off", "📊 Graph Topology & Stats"])

with tab1:
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("Query Configuration")
        k_val = st.slider("Top-k Nearest Neighbors", min_value=1, max_value=30, value=10)
        filter_opt = st.selectbox("Metadata Filter (`where`)", ["None", "category == 'technical'", "category == 'general'"])
        where_filter = None
        if filter_opt == "category == 'technical'":
            where_filter = {"category": "technical"}
        elif filter_opt == "category == 'general'":
            where_filter = {"category": "general"}

        query_source = st.radio("Query Source", ["Select Existing Node", "Random Synthetic Vector"], index=0)
        if query_source == "Select Existing Node":
            target_idx = st.number_input("Target Node Index", min_value=0, max_value=len(vectors) - 1, value=0)
            query_vec = vectors[target_idx]
        else:
            rng_q = np.random.default_rng(999)
            query_vec = rng_q.normal(size=dimension).astype(np.float32)
            if metric == "cosine":
                query_vec = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)

        # Run Search
        t_flat_0 = time.perf_counter()
        exact_results = flat_idx.search(query_vec, k=k_val, where=where_filter)
        t_flat = (time.perf_counter() - t_flat_0) * 1000.0

        t_hnsw_0 = time.perf_counter()
        approx_results = hnsw_idx.search(query_vec, k=k_val, ef_search=param_ef_s, where=where_filter)
        t_hnsw = (time.perf_counter() - t_hnsw_0) * 1000.0

        exact_ids = [vid for vid, _ in exact_results]
        approx_ids = [vid for vid, _ in approx_results]
        overlap = set(exact_ids) & set(approx_ids)
        recall_k = len(overlap) / max(len(exact_ids), 1)

        st.markdown(f"**Measured Recall@{k_val}:** `{recall_k:.1%}`")
        st.markdown(f"- **Flat Exact Latency:** `{t_flat:.3f} ms`")
        st.markdown(f"- **HNSW ANN Latency:** `{t_hnsw:.3f} ms`")

    with col_right:
        # Project Query onto 2D PCA space
        centered_q = query_vec - np.mean(vectors, axis=0)
        _, _, vt = np.linalg.svd(vectors - np.mean(vectors, axis=0), full_matrices=False)
        q_2d = centered_q @ vt[:2].T

        # Create Scatter Plot
        fig = go.Figure()

        # Background points
        fig.add_trace(go.Scatter(
            x=pca_coords[:, 0],
            y=pca_coords[:, 1],
            mode="markers",
            marker=dict(size=5, color="#334155", opacity=0.5),
            name="Indexed Vectors",
            hoverinfo="none",
        ))

        # Exact Ground Truth Neighbors
        exact_indices = [int(vid.split("_")[1]) for vid in exact_ids]
        fig.add_trace(go.Scatter(
            x=pca_coords[exact_indices, 0],
            y=pca_coords[exact_indices, 1],
            mode="markers",
            marker=dict(size=12, color="#34d399", symbol="circle-open", line=dict(width=2)),
            name=f"Exact Top-{k_val} (Ground Truth)",
            text=exact_ids,
            hoverinfo="text",
        ))

        # HNSW Retrieved Neighbors
        approx_indices = [int(vid.split("_")[1]) for vid in approx_ids]
        fig.add_trace(go.Scatter(
            x=pca_coords[approx_indices, 0],
            y=pca_coords[approx_indices, 1],
            mode="markers",
            marker=dict(size=8, color="#38bdf8", symbol="cross"),
            name=f"HNSW Top-{k_val} (Approximate)",
            text=approx_ids,
            hoverinfo="text",
        ))

        # Query Vector
        fig.add_trace(go.Scatter(
            x=[q_2d[0]],
            y=[q_2d[1]],
            mode="markers",
            marker=dict(size=16, color="#f59e0b", symbol="star"),
            name="Query Vector (q)",
            hoverinfo="name",
        ))

        fig.update_layout(
            title=f"2D PCA Projection: Approximate vs. Exact Search (Recall: {recall_k:.1%})",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0b0f19",
            font=dict(color="#94a3b8"),
            xaxis=dict(showgrid=True, gridcolor="#1e293b", zerolinecolor="#334155"),
            yaxis=dict(showgrid=True, gridcolor="#1e293b", zerolinecolor="#334155"),
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Recall vs. QPS Trade-Off Curve (Parameter efSearch)")
    st.write("Examines how query depth (`efSearch`) increases recall at the cost of queries per second.")

    if st.button("⚡ Run Interactive efSearch Sweep"):
        with st.spinner("Executing test queries across efSearch values..."):
            rng_bench = np.random.default_rng(123)
            test_queries = rng_bench.normal(size=(50, dimension)).astype(np.float32)
            if metric == "cosine":
                test_queries = test_queries / np.maximum(np.linalg.norm(test_queries, axis=1, keepdims=True), 1e-12)

            sweep_efs = [8, 16, 32, 64, 128, 256]
            recalls = []
            qps_vals = []
            p50_lats = []

            # Ground truth for test set
            gt_set = []
            for tq in test_queries:
                gt_set.append({item[0] for item in flat_idx.search(tq, k=10)})

            for cur_ef in sweep_efs:
                lat_list = []
                rec_sum = 0.0
                t_total_0 = time.perf_counter()

                for q_i, tq in enumerate(test_queries):
                    t0 = time.perf_counter()
                    res = hnsw_idx.search(tq, k=10, ef_search=cur_ef)
                    lat_list.append((time.perf_counter() - t0) * 1000.0)

                    approx_set = {item[0] for item in res}
                    rec_sum += len(approx_set & gt_set[q_i]) / 10.0

                total_dur = time.perf_counter() - t_total_0
                recalls.append(rec_sum / len(test_queries))
                qps_vals.append(len(test_queries) / max(total_dur, 1e-6))
                lat_list.sort()
                p50_lats.append(float(np.percentile(lat_list, 50)))

            # Plot Trade-off
            fig_tradeoff = go.Figure()
            fig_tradeoff.add_trace(go.Scatter(
                x=qps_vals,
                y=recalls,
                mode="lines+markers+text",
                text=[f"ef={ef}" for ef in sweep_efs],
                textposition="top center",
                line=dict(color="#38bdf8", width=3),
                marker=dict(size=10, color="#f43f5e"),
            ))

            fig_tradeoff.update_layout(
                title="Pareto Frontier: Recall@10 vs. Throughput (QPS)",
                xaxis_title="Throughput (QPS) — Higher is better",
                yaxis_title="Recall@10 — Higher is better",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0b0f19",
                font=dict(color="#94a3b8"),
                xaxis=dict(showgrid=True, gridcolor="#1e293b"),
                yaxis=dict(showgrid=True, gridcolor="#1e293b", range=[0.5, 1.05]),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_tradeoff, use_container_width=True)

with tab3:
    st.subheader("HNSW Graph Topology & Memory Profile")
    stats_data = hnsw_idx.stats()
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.json(stats_data)
    with c_s2:
        layer_dist = stats_data["layer_distribution"]
        fig_layers = go.Figure(go.Bar(
            x=[f"Layer {lvl}" for lvl in sorted(layer_dist.keys())],
            y=[layer_dist[lvl] for lvl in sorted(layer_dist.keys())],
            marker_color="#38bdf8",
        ))
        fig_layers.update_layout(
            title="Vector Distribution by Maximum Layer Level",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0b0f19",
            font=dict(color="#94a3b8"),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_layers, use_container_width=True)
