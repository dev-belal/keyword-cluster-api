# Keyword Clustering API - Flask Application

## Overview
A Python Flask API service that performs keyword clustering using machine learning. The API accepts keywords with their embeddings and groups them based on cosine similarity using agglomerative clustering.

## Project Structure
```
.
├── main.py           # Main Flask application with clustering API
├── requirements.txt  # Python dependencies
└── .gitignore       # Git ignore file for Python projects
```

## Recent Changes
- **November 11, 2025**: Updated to keyword clustering API
  - Replaced simple Flask app with full clustering API implementation
  - Added flask-cors for CORS support
  - Added numpy for numerical operations
  - Added scikit-learn for machine learning clustering
  - Configured API key authentication (x-api-key header)
  - Set default port to 5000 to match workflow configuration
  - Added safety limits for embeddings and keywords

## Dependencies
- Flask==3.0.0 - Web framework
- flask-cors==4.0.0 - CORS support
- numpy==1.26.0 - Numerical computing
- scikit-learn==1.3.2 - Machine learning library

## Running the Application
The Flask app runs automatically via the configured workflow:
- Workflow name: flask-app
- Command: python main.py
- Server: http://0.0.0.0:5000
- Debug mode: enabled

## API Endpoints

### GET /
Health check endpoint that returns:
```json
{"ok": true, "msg": "Keyword Clustering API ready"}
```

### POST /cluster
Clusters keywords based on their embeddings.

**Authentication**: Requires `x-api-key` header (default: "dev-key")

**Request Body**:
```json
{
  "keywords": ["keyword1", "keyword2", ...],
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
  "threshold": 0.78,
  "max_clusters": null
}
```

**Response**: Dictionary of clusters (label -> list of keywords)

## Environment Variables
- `CLUSTER_API_KEY` - API key for authentication (default: "dev-key")
- `MAX_EMBEDDINGS` - Maximum embeddings allowed (default: 2000)
- `MAX_KEYWORDS_PER_REQUEST` - Maximum keywords per request (default: 2000)
- `PORT` - Port to run the server on (default: 5000)

## Features
- API key authentication via x-api-key header
- Keyword clustering using agglomerative clustering
- Cosine similarity-based distance calculation
- Configurable similarity threshold
- Safety limits to prevent OOM errors
- CORS enabled for cross-origin requests