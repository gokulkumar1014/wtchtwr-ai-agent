# Qdrant Local Storage (qdrant_storage/)

This directory stores the local on-disk Qdrant index used by wtchtwr for semantic review search (RAG).  
These files are auto-generated, machine-specific, and must not be committed to Git.

---

## What Should NOT Be Committed

Everything inside:

```
qdrant_storage/
├── aliases/
├── collections/
├── .deleted/
└── raft_state.json
```

These include:

- HNSW vector index files  
- payload blobs  
- shard/segment metadata  
- internal raft state  
- absolute machine paths  

They change frequently and cannot be shared across machines.

---

## What Is Committed

Only:

```
qdrant_storage/
└── .gitkeep
└── README.md
```

This keeps the directory present in the repo without storing binary files.

---

## How This Folder Gets Populated

This folder is automatically populated when you run:

```
python scripts/reindex_qdrant_reviews.py
```

The script:

1. Loads embeddings and metadata from:
   - vec/airbnb_reviews/reviews_embeddings.npy
   - vec/airbnb_reviews/reviews_metadata.parquet

2. Deletes old Qdrant collection  
3. Recreates the `airbnb_reviews` collection  
4. Uploads ~650K vectors  
5. Qdrant writes new files automatically into qdrant_storage/collections/

---

# Team Setup Guide (Required for All Members)

Every teammate must follow these steps to run RAG correctly.

---

## 1. Install Docker Desktop (Required)

### Windows 10/11
1. Install Docker Desktop:  
   https://www.docker.com/products/docker-desktop/
2. Enable WSL2 when asked.
3. Restart your machine.

### macOS (Intel or M1/M2)
1. Download Docker Desktop:  
   https://www.docker.com/products/docker-desktop/
2. Move it to Applications.

### Linux (Ubuntu)
```
sudo apt install docker.io
sudo systemctl enable --now docker
```

Verify Docker works:
```
docker --version
docker run hello-world
```

If you see a success message, Docker is ready.

---

## 2. Clone the Repository

```
git clone https://github.com/gokulkumar1014/wtchtwr-AI-Powered-Property-Performance-and-Market-Insights-Agent.git wtchtwr-agent
cd wtchtwr-agent
```

---

## 3. Download Required Vector Files

Download the following two files from Google Drive:

- reviews_embeddings.npy  
- reviews_metadata.parquet  

Place them here:

```
vec/airbnb_reviews/
    reviews_embeddings.npy
    reviews_metadata.parquet
```

If the folder does not exist, create it manually.

---

## 4. Start Qdrant Using Docker

### First-time setup (run only once):
```
docker run -d \
  --name hope-qdrant \
  -p 6333:6333 \
  qdrant/qdrant
```

### After first time (every day):
Start Qdrant:
```
docker start hope-qdrant
```

Stop Qdrant:
```
docker stop hope-qdrant
```

### Verify Qdrant is running:

Open:
```
http://localhost:6333/dashboard
```

You should eventually see:

- airbnb_reviews collection  
- Status: green  
- Points: ~650K  

---

## 5. Rebuild the Local Qdrant Collection

Run:

```
python scripts/reindex_qdrant_reviews.py
```

This will:

- delete old collection  
- recreate `airbnb_reviews`  
- upload all vectors  
- perform verification queries  
- write new files into qdrant_storage/

Expected logs:

```
Upserted 512/657704
Upserted 1024/657704
...
Verification listing_id hits: 5
Reindex complete — 657704 points ingested
```

---

## 6. Run Backend

```
uvicorn backend.app:app --reload
```

Backend is available at:
```
http://localhost:8000
```

---

## 7. Run Frontend

```
cd frontend
npm install
npm run dev
```

Frontend opens at:
```
http://localhost:5173
```

You must have Qdrant running first.

---

## 8. Common Issues

### Docker not found
Docker Desktop is not installed or not running.

### Cannot connect to Qdrant
Run:
```
docker start hope-qdrant
```

### Missing vector files
Ensure:
```
vec/airbnb_reviews/reviews_embeddings.npy
vec/airbnb_reviews/reviews_metadata.parquet
```

### Embedding count mismatch
Re-download both files.

### Qdrant shows 0 points
Run:
```
python scripts/reindex_qdrant_reviews.py
```

---

## Summary

- `qdrant_storage/` is never version-controlled  
- It is always generated locally  
- The real source of truth is:
  - reviews_embeddings.npy
  - reviews_metadata.parquet
- The indexing script rebuilds everything identically across all teammates
