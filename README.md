# 🏠 wtchtwr — AI-Powered Property Performance & Market Insights Agent

**wtchtwr** is an AI-driven analytics assistant built for short-term rental (STR) operators, property managers, analysts, and data teams.

It combines:

- **Agentic AI (LangGraph)**
- **Hybrid SQL + RAG Retrieval**
- **Qdrant Vector Search**
- **DuckDB Analytics Engine**
- **React + Vite Frontend**
- **Deterministic Multi-Agent Workflows**

wtchtwr powers:
- 📊 **Portfolio Triage** — detect low performers, winners, pricing gaps  
- 🧭 **Expansion Scout** — evaluate high‑growth neighborhoods  
- 🧮 **Hybrid SQL/RAG Chat** — structured + unstructured insights  
- 🔍 **Review Intelligence** — sentiment/RAG over 650K+ Airbnb reviews  

---

# 📦 Full Project Architecture

```
wtchtwr/
├── agent/                  # LangGraph graph, nodes, policy, vector logic
├── backend/                # FastAPI backend
├── frontend/               # React + Vite UI
├── config/                 # environment configs, keys, service accounts
├── db/                     # DuckDB file (airbnb.duckdb)
├── vec/                    # Embeddings + metadata (required)
├── qdrant_storage/         # Local Qdrant index (NEVER committed)
├── scripts/                # Reindex, utilities
├── tests/                  # Unit + integration tests
├── .env.example            # Template environment file
└── README.md               # This file
```

---

# 🚀 Features (High Level)

## 1. **Portfolio Triage (AI-driven)**
- Identifies:
  - Low-occupancy listings  
  - Complaint-heavy units  
  - Underpriced winners  
  - Optimization opportunities  
- Combines:
  - DuckDB metrics  
  - RAG snippets  
  - Metadata filters  
  - Structured + unstructured signals  

## 2. **Expansion Scout**
- Neighborhood-level analytics  
- Comp set comparisons  
- Seasonality scoring  
- Investment attractiveness  

## 3. **Review Intelligence (RAG)**
- 657,704 Airbnb reviews embedded using **MiniLM-L6-v2**
- Qdrant local index powering:
  - Complaint clustering  
  - Sentiment lookup  
  - Root-cause discovery  

## 4. **Hybrid SQL + RAG Chat**
- LangGraph pipeline:
  ```
  classify_intent → resolve_entities → plan_steps → SQL/RAG path → execute → compose
  ```
- Deterministic + debuggable  
- No hallucinations (SQL is strict)  

---

### 🧠 Core Dependencies

- ![Python](https://img.shields.io/badge/Python-3.11-blue.svg) **Python 3.11**
- ![Node](https://img.shields.io/badge/Node.js-18+-green.svg) **Node.js 18+**
- ![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg) **FastAPI (Backend API)**
- ![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Orchestration-purple.svg) **LangGraph**
- ![LangChain](https://img.shields.io/badge/LangChain-NLP%20Pipelines-2e6ac7.svg) **LangChain**
- ![OpenAI](https://img.shields.io/badge/OpenAI-LLM%20Models-412991.svg) **OpenAI API**
- ![SentenceTransformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-ff6f00.svg) **Sentence Transformers**
- ![DuckDB](https://img.shields.io/badge/DuckDB-Analytical%20DB-yellow.svg) **DuckDB 1.4+**
- ![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Database-orange.svg) **Qdrant Vector DB**
- ![React](https://img.shields.io/badge/React-Frontend-61DAFB.svg) **React (Frontend)**
- ![Vite](https://img.shields.io/badge/Vite-Bundler-9370DB.svg) **Vite**
- ![Tailwind](https://img.shields.io/badge/TailwindCSS-Styling-38B2AC.svg) **TailwindCSS**
- ![Tavily](https://img.shields.io/badge/Tavily-Search%20API-black.svg) **Tavily Search API**
- ![Slack](https://img.shields.io/badge/Slack-Bot%20Integration-4A154B.svg) **Slack Bot Integration**
- ![Parquet](https://img.shields.io/badge/Parquet-Data%20Storage-0A66C2.svg) **Parquet / Arrow**
- ![Pandas](https://img.shields.io/badge/Pandas-Data%20Frames-150458.svg) **Pandas**
- ![Numpy](https://img.shields.io/badge/Numpy-Arrays-013243.svg) **NumPy**
- ![ScikitLearn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg) **scikit-learn**
- ![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI%20Server-ff44aa.svg) **Uvicorn**

---

# ⚙️ Installation & Setup (For Teammates)

## 1️⃣ Install Docker
Recommended — Qdrant runs perfectly through Docker.

Verify:
```
docker --version
docker run hello-world
```

## 2️⃣ Clone the Repository
```
git clone https://github.com/gokulkumar1014/wtchtwr-ai-agent.git
cd wtchtwr-ai-agent
```

## 3️⃣ Create Virtual Environment
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 4️⃣ Download Vector Assets
Place inside:

```
vec/airbnb_reviews/reviews_embeddings.npy
vec/airbnb_reviews/reviews_metadata.parquet
```

## 5️⃣ Start Qdrant (Every Time You Use the App)
First time:
```
docker run -d --name hope-qdrant -p 6333:6333 qdrant/qdrant
```
Daily:
```
docker start hope-qdrant
```

Access dashboard:
```
http://localhost:6333/dashboard
```

## 6️⃣ Reindex Reviews (Only first time)
```
python scripts/reindex_qdrant_reviews.py
```

## 7️⃣ Start Backend
```
uvicorn backend.main:app --reload --port 8000
```

## 8️⃣ Start Frontend
In new terminal 
```
cd frontend
npm install
npm run dev
```

Frontend opens at:
```
http://localhost:5173
```

---

# 🧩 Important Notes

## ❗ qdrant_storage/ is NEVER committed
It is always generated locally by Qdrant.  

## ❗ vec/ must contain real embeddings
These are mandatory:
- `reviews_embeddings.npy`
- `reviews_metadata.parquet`

## ❗ .env must be copied from .env.example
You must fill:
- `OPENAI_API_KEY`
- Slack configs (optional)
- Gmail app password (optional)  

---

# 🧪 Testing
```
pytest -q
```

---

# 🛠️ Scripts

```
scripts/
└── reindex_qdrant_reviews.py
└── load_duckdb.py
```

Purpose:
- wipe old index
- rebuild Qdrant
- upload all 657K vectors
- validate index health

---

# 🧭 Development Workflow

1. Update code in `agent/` or `backend/`  
2. Restart backend  
3. Test query using frontend  
4. Validate RAG / SQL behavior  
5. Iterate  

---

# 🏁 Summary

wtchtwr is a fully agentic STR intelligence system providing:

- 🔍 Deep review insights  
- 📊 Portfolio optimization  
- 🧠 Multi-agent structured reasoning  
- 🗂️ SQL + RAG hybrid analytics  
- 🧭 Neighborhood scouting  
- 🖥️ Clean, modular architecture  
- 🧲 Deterministic pipelines (LangGraph)

This README serves as the **official onboarding + setup guide** for you and your teammates.

---

© 2025 wtchtwr — Built with ❤️ by Gokul
