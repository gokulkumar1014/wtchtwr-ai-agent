# 📦 Airbnb Reviews Vector Artifacts (`vec/airbnb_reviews/`)

This directory stores **precomputed embeddings and metadata** used to build and serve the Airbnb reviews vector store for wtchtwr's RAG pipeline.  
These files are **large binary artifacts**, so they are *not committed* to Git.  
Instead, teams download them from Google Drive or regenerate them using the project scripts.

---

## 📁 Directory Structure

vec/
└── airbnb_reviews/
├── reviews_embeddings.npy # 384-dim MiniLM embedding matrix
├── reviews_metadata.parquet # Review metadata + sentiment payloads
├── .gitkeep # Ensures folder persists in Git
└── README.md

### Artifact Descriptions

| File | Description |
|------|-------------|
| **`reviews_embeddings.npy`** | NumPy matrix of normalized embeddings generated using `sentence-transformers/all-MiniLM-L6-v2`. |
| **`reviews_metadata.parquet`** | DataFrame containing listing_id, comment_id, month/year, sentiment scores, neighbourhood, text, and Highbury flag. |
| **`.gitkeep`** | Placeholder file so the directory stays tracked. |

These files fuel the **Qdrant vector store**, powering review-based semantic search and hybrid SQL+RAG insights.

---

## 📥 Download Prebuilt Artifacts (Recommended)

To avoid regenerating embeddings, use the ready-to-load artifact bundle:

🔗 **Google Drive Folder:**  
https://drive.google.com/drive/u/1/folders/1cw9CvJ76yWudlL-0UWsAs94pM5JBHboa

Place the downloaded files into:

vec/airbnb_reviews/

This enables instant Qdrant bootstrapping with no local computation.

---

## 🔄 Rebuild Locally From Scratch (Developer Workflow)

If you have the cleaned sentiment-scored dataset:
 data/clean/reviews_sentiment_scores.parquet

you can regenerate embeddings + metadata and rebuild the Qdrant collection.

### 1. Activate your environment
```bash
source .venv/bin/activate
2. Run the rebuild script
bash
Copy code
python scripts/rebuild_review_vectors.py

What this script does:
1. Loads enriched sentiment review data
2. Generates MiniLM (384-dim) embeddings
3. Writes artifacts to vec/airbnb_reviews/
4. Deletes + recreates the Qdrant collection
5. Uploads all vectors + metadata in batches

To reuse existing artifacts and only rebuild the Qdrant index:

python scripts/rebuild_review_vectors.py --reuse-artifacts


🧠 Notes for Contributors

If switching embedding models, clear Qdrant storage:
rm -rf qdrant_storage/collections/<collection_name>

🧪 Verification Checklist
After rebuilding, logs should include something like:

Rows processed: 88,531
Embeddings shape: (88531, 384)
Sentiment label counts:
  positive: 42,xxx
  neutral: 27,xxx
  negative: 18,xxx
Qdrant upload complete: collection 'airbnb_reviews' now contains 88,531 points

This confirms:
Metadata parsed correctly
Embeddings match metadata row count
Qdrant populated with all points

✅ Summary
This folder holds all vector artifacts needed for RAG, while keeping the Git repo lightweight.
You can either:
1. Download the artifacts (fastest), or
2. Regenerate them with the rebuild script (reproducible)

Both workflows are fully supported by wtchtwr.
