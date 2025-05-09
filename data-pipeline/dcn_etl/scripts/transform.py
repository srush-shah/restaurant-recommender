import pandas as pd
from pathlib import Path

REVIEWS_PATH = "/staging/raw/yelp_academic_dataset_review.json"
BUSINESS_PATH = "/staging/raw/yelp_academic_dataset_business.json"
USER_VEC_PATH = "/staging/user_latent_vectors/part-00000-b2130b85-767f-429d-b9ac-34aef309bdf8-c000.csv"
ITEM_VEC_PATH = "/staging/item_latent_vectors/part-00000-939a5f7d-27f3-4b2b-b4ce-13e0147ec1ab-c000.csv"
OUT_CSV = Path("/staging/transformed.csv")

# 1) Load the smaller tables
print("Loading business data and filtering restaurants…")
business = (
    pd.read_json(BUSINESS_PATH, lines=True)
      .loc[:, ["business_id", "attributes", "categories"]]
)
# keep only restaurants
restaurant_business = business[
    business["categories"].str.contains("Restaurant", case=False, na=False)
]

print("Loading user latent vectors…")

user_vec = (
    pd.concat(
      [pd.read_csv(p) for p in sorted(Path("/staging/user_latent_vectors").glob("*.csv"))],
      ignore_index=True
    )
    .loc[:, ["user_id", "features"]]
    .rename(columns={"features": "user_latent_vector"})
)

print("Loading item latent vectors…")
item_vec = (
    pd.concat(
      [pd.read_csv(p) for p in sorted(Path("/staging/item_latent_vectors").glob("*.csv"))],
      ignore_index=True
    )
    .rename(columns={"features": "item_latent_vector"})
    .loc[:, ["business_id", "item_latent_vector"]]
)

# Prepare output file
if OUT_CSV.exists():
    OUT_CSV.unlink()
print(f"Writing output to {OUT_CSV}")

# 2) Process reviews in chunks
print("Loading reviews data…")
chunk_iter = pd.read_json(REVIEWS_PATH, lines=True, chunksize=100_000)

for i, reviews_chunk in enumerate(chunk_iter):
    print(f"  Processing chunk {i+1}…")

    reviews = (
        reviews_chunk
          .loc[:, ["review_id", "user_id", "business_id", "text", "stars"]]
          .rename(columns={"text": "review_text", "stars": "rating"})
    )

    # restrict to restaurant reviews only
    reviews = reviews[reviews["business_id"].isin(restaurant_business["business_id"])]

    # Merge business info
    merged = reviews.merge(
        restaurant_business,
        on="business_id",
        how="inner"
    )

    # Merge in user & item vectors
    merged = (
      merged
        .merge(user_vec, on="user_id", how="left")
        .merge(item_vec, on="business_id", how="left")
    )

    # Append to CSV
    mode = "w" if i == 0 else "a"
    header = (i == 0)
    merged.to_csv(OUT_CSV, mode=mode, header=header, index=False)


print("All chunks done.")
print(f"Final rows: {sum(1 for _ in open(OUT_CSV)) - 1}")


