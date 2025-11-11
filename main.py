# app.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import gradio as gr
import typing as t

def cluster_payload(payload: dict) -> dict:
    """
    Expected payload shape:
      {
        "keywords": ["kw1","kw2",...],
        "embeddings": [[...], [...], ...],
        "threshold": 0.78    # optional, default 0.78
      }
    Returns:
      {
        "clusters": {
            "0": ["kwA", "kwB"],
            "1": ["kwC", "kwD", ...],
            ...
        },
        "labels": [0,0,1,1,...]   # label per keyword
      }
    """
    # basic validation
    keywords = payload.get("keywords")
    embeddings = payload.get("embeddings")
    threshold = float(payload.get("threshold", 0.78))

    if not keywords or not embeddings:
        return {"error": "payload must include 'keywords' and 'embeddings' arrays."}

    # convert to numpy array
    X = np.array(embeddings, dtype=float)
    if X.ndim != 2:
        return {"error": "embeddings must be a 2D array: list of vectors."}
    if len(keywords) != X.shape[0]:
        return {"error": "length of 'keywords' must match number of 'embeddings' rows."}

    # compute cosine similarity matrix (N x N)
    # numeric stability: if vectors are all zero, cosine_similarity may produce NaN
    try:
        sim = cosine_similarity(X)
    except Exception as e:
        return {"error": f"cosine similarity failure: {str(e)}"}

    # convert to distance matrix for clustering
    # distance = 1 - similarity (range ~ [0,2], but typical [0,1])
    dist = 1.0 - sim

    # AgglomerativeClustering with precomputed distances
    # we set n_clusters=None and distance_threshold to split based on threshold
    # scikit-learn expects a symmetric matrix; we pass 'precomputed'
    try:
        # distance_threshold = 1 - similarity_threshold
        distance_threshold = max(0.0, 1.0 - threshold)
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold
        )
        labels = model.fit_predict(dist)
    except TypeError:
        # scikit-learn older/newer versions may use metric instead of affinity
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold
        )
        labels = model.fit_predict(dist)

    # group keywords by label
    clusters = {}
    for kw, lbl in zip(keywords, labels):
        clusters.setdefault(str(int(lbl)), []).append(kw)

    return {
        "clusters": clusters,
        "labels": [int(x) for x in labels]
    }

# Gradio interface: a single JSON in/out component so the space exposes /api/predict
with gr.Blocks() as demo:
    gr.Markdown("## Keyword Clustering API — send a JSON payload to this Space's /api/predict")
    input_json = gr.JSON(label="Payload JSON")
    output_json = gr.JSON(label="Clusters")
    btn = gr.Button("Run (for manual testing)")
    btn.click(fn=cluster_payload, inputs=input_json, outputs=output_json)

# `demo` app will be used by HF Spaces; below allows running locally with `python app.py`
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
