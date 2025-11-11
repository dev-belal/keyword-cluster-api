# main.py
import os
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Security: require an API key via header "x-api-key"
EXPECTED_API_KEY = os.environ.get("CLUSTER_API_KEY", "dev-key")

# Safety limits
MAX_EMBEDDINGS = int(os.environ.get("MAX_EMBEDDINGS", 2000))  # to avoid OOM
MAX_KEYWORDS_PER_REQUEST = int(os.environ.get("MAX_KEYWORDS_PER_REQUEST", 2000))

@app.route("/", methods=["GET"])
def root():
    return jsonify({"ok": True, "msg": "Keyword Clustering API ready"})

@app.route("/cluster", methods=["POST"])
def cluster():
    # Basic auth
    key = request.headers.get("x-api-key")
    if key is None or key != EXPECTED_API_KEY:
        return abort(401, "Missing or invalid API key")

    data = request.json
    if not data:
        return abort(400, "Expected JSON body")

    keywords = data.get("keywords")
    embeddings = data.get("embeddings")
    threshold = float(data.get("threshold", 0.78))
    max_clusters = data.get("max_clusters")  # optional

    if not isinstance(keywords, list) or not isinstance(embeddings, list):
        return abort(400, "keywords and embeddings must be lists")

    if len(keywords) != len(embeddings):
        return abort(400, "keywords and embeddings must have same length")

    n = len(keywords)
    if n == 0:
        return jsonify({})

    if n > MAX_KEYWORDS_PER_REQUEST:
        return abort(400, f"Too many items in request (max {MAX_KEYWORDS_PER_REQUEST})")

    # Convert embeddings to numpy array
    try:
        X = np.array(embeddings, dtype=float)
    except Exception as e:
        return abort(400, f"Invalid embeddings format: {str(e)}")

    # If only 1 item, return single cluster
    if n == 1:
        return jsonify({"0": [keywords[0]]})

    # Compute cosine similarity matrix (n x n)
    sim = cosine_similarity(X)
    # Convert to distance matrix for clustering
    dist = 1.0 - sim

    # Agglomerative clustering with a distance threshold:
    # clusters where (1 - cosine_similarity) <= (1 - threshold)  => similarity >= threshold
    distance_threshold = 1.0 - threshold
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold
        )
        labels = model.fit_predict(dist)
    except TypeError:
        # Older sklearn versions may not accept "affinity"; try metric param
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold
        )
        labels = model.fit_predict(dist)

    # Build clusters dict: label -> list of keywords
    clusters = {}
    for kw, label in zip(keywords, labels):
        clusters.setdefault(str(int(label)), []).append(kw)

    # Optionally sort clusters by size descending
    sorted_clusters = dict(sorted(clusters.items(), key=lambda kv: -len(kv[1])))

    return jsonify(sorted_clusters)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
