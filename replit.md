# Flask Application

## Overview
A basic Python Flask web application with a simple "Hello, Flask!" route.

## Project Structure
```
.
├── main.py           # Main Flask application file
├── requirements.txt  # Python dependencies (Flask==3.0.0)
└── .gitignore       # Git ignore file for Python projects
```

## Recent Changes
- **November 11, 2025**: Initial project setup
  - Created main.py with basic Flask application
  - Created requirements.txt with Flask 3.0.0 dependency
  - Installed Python 3.11 and Flask
  - Configured flask-app workflow running on port 5000
  - Added .gitignore for Python projects

## Running the Application
The Flask app runs automatically via the configured workflow:
- Workflow name: flask-app
- Command: python main.py
- Server: http://0.0.0.0:5000

## Features
- Single route at "/" that returns "Hello, Flask!"
- Configured to run on all addresses (0.0.0.0) on port 5000