import pandas as pd

# --- 1. Load file ---
path = "/home/gokul_kumar_kesavan/airbnb-agent/data/clean/review_sentiment_scores.parquet"  # change path if needed
df = pd.read_parquet(path)

# --- 2. Basic info ---
print("\n=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS & DTYPES ===")
print(df.dtypes)

print("\n=== FIRST 10 ROWS ===")
print(df.head(10))

# --- 3. Sentiment distribution ---
print("\n=== SENTIMENT LABEL COUNTS ===")
print(df['sentiment_label'].value_counts(dropna=False))

print("\n=== SENTIMENT LABEL PERCENTAGES ===")
print(df['sentiment_label'].value_counts(normalize=True).round(4))

# --- 4. Listing-level diagnostics ---
print("\n=== UNIQUE LISTINGS ===")
print(df['listing_id'].nunique())

print("\n=== SAMPLE LISTING IDS ===")
print(df['listing_id'].dropna().unique()[:20])

# --- 5. Time coverage ---
print("\n=== UNIQUE YEARS ===")
print(sorted(df['year'].dropna().unique()))

print("\n=== UNIQUE MONTHS ===")
print(sorted(df['month'].dropna().unique()))

# --- 6. Missing values summary ---
print("\n=== NULL COUNTS ===")
print(df.isna().sum())

# --- 7. Stats on VADER scores ---
numeric_cols = ['negative', 'neutral', 'positive', 'compound']
print("\n=== VADER SCORE STATS ===")
print(df[numeric_cols].describe().round(4))


# ===========================================================
# === 8. Inspect a specific row / listing / comment ID ======
# ===========================================================

# --- Pick one specific listing_id ---
listing_to_view = 2595   # <-- change this to inspect any listing
subset_listing = df[df['listing_id'] == listing_to_view]

print(f"\n=== ALL REVIEWS FOR LISTING {listing_to_view} ===")
print(subset_listing.head(20))  # print first 20 reviews

# --- Pick one specific comment_id ---
comment_to_view = int(subset_listing.iloc[0]['comment_id']) if not subset_listing.empty else None
if comment_to_view:
    row = df[df['comment_id'] == comment_to_view].iloc[0]
    print(f"\n=== FULL DETAIL FOR COMMENT_ID {comment_to_view} ===")
    print(row)

# --- Random row (to inspect arbitrary review) ---
print("\n=== RANDOM ROW ===")
print(df.sample(1).iloc[0])

# --- First row (clean check) ---
print("\n=== FIRST ROW ===")
print(df.iloc[0])