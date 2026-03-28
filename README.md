# 🏀 NBA Oracle: Hybrid AI MLOps Pipeline
![Python](https://img.shields.io/badge/python-3.9-blue.svg) ![Docker](https://img.shields.io/badge/docker-containerized-blue.svg) ![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD-green.svg)

An enterprise-grade, end-to-end Machine Learning pipeline that predicts NBA clutch performance and game momentum using a dual-model AI architecture.

## 🧠 System Architecture
This project implements an Ensemble Multi-Model Gateway:
1. **XGBoost Regressor (Clutch Projection):** Analyzes historical feature-engineered data (Assist-to-Turnover ratios, Per-Minute Production) to project a player's mathematical Plus/Minus impact in the final 5 minutes of games.
2. **PyTorch Transformer (Momentum Analysis):** A multi-head attention NLP architecture that reads chronological play-by-play events to predict the next likely game sequence.

## 🛠️ Tech Stack
- **Data Engineering:** `nba_api`, Pandas, PyArrow (Custom exponential backoff scraping to bypass Akamai firewalls).
- **Machine Learning:** PyTorch, XGBoost, Scikit-Learn.
- **Backend API:** FastAPI & Uvicorn.
- **Frontend UI:** Streamlit & Plotly (for comparative Radar Charts).
- **DevOps:** Docker Compose, GitHub Actions (Automated Weekly Retraining).

## 🚀 Local Deployment
This project is fully containerized. To run the API and UI locally:
\`\`\`bash
docker compose up --build
\`\`\`
Access the dashboard at: `http://localhost:8501`

## 🔄 CI/CD Pipeline
Included in `.github/workflows/mlops_pipeline.yml` is an automated GitHub Action that triggers every Sunday at midnight. It spins up an Ubuntu runner, updates dependencies, extracts the latest week of NBA statistics, retrains the XGBoost model, and commits the fresh weights back to the repository.
<img width="1815" height="811" alt="{3DDD9461-4967-4B7E-9307-97E8C98CB408}" src="https://github.com/user-attachments/assets/7b3a9ef8-9455-4124-9c37-b35336f9f988" />
