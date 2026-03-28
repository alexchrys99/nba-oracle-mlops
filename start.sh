#!/bin/bash

# Start the FastAPI backend in the background
echo "🚀 Starting FastAPI Backend..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for the API to boot up
sleep 5

# Start the Streamlit UI
echo "🖥️ Starting Streamlit Frontend..."
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
