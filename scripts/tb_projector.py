import os
import numpy as np
import tensorflow as tf
import csv
from langchain_chroma import Chroma

# -----------------------------
# Load your Chroma collection
# -----------------------------
PERSIST_DIR = "vec/airbnb_reviews"
vs = Chroma(collection_name="airbnb_nyc_reviews",
            persist_directory=PERSIST_DIR)

data = vs.get()
ids = data["ids"]
metas = data["metadatas"]
texts = data["documents"]

print(f"Loaded {len(ids)} docs from Chroma")

# -----------------------------
# 1) Save embeddings as TensorFlow checkpoint
# -----------------------------
embs = np.array(vs._collection.get(include=["embeddings"])["embeddings"])
print("Shape of embeddings:", embs.shape)

logdir = "logs/projector"
os.makedirs(logdir, exist_ok=True)

ckpt = tf.Variable(embs, name="embedding")
ckpt_saver = tf.train.Checkpoint(embedding=ckpt)
ckpt_saver.save(os.path.join(logdir, "emb.ckpt"))

# -----------------------------
# 2) Save metadata TSV
# -----------------------------
meta_path = os.path.join(logdir, "projector_metadata.tsv")
with open(meta_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["id", "year", "reviewer_name", "text"])
    for i, meta in enumerate(metas):
        writer.writerow([
            ids[i],
            meta.get("year", ""),
            meta.get("reviewer_name", ""),
            (texts[i] or "")[:100].replace("\n", " ")
        ])
print(f"Metadata saved to {meta_path}")

# -----------------------------
# 3) Configure projector
# -----------------------------
from tensorboard.plugins import projector

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = "projector_metadata.tsv"

projector.visualize_embeddings(logdir, config)

print("✅ Now run: tensorboard --logdir logs/projector")
