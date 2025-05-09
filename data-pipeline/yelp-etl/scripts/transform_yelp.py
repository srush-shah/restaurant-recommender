import pandas as pd
import os, json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load business data
business_df = pd.read_json("data/raw/yelp_academic_dataset_business.json", lines=True)

# Create business city mapping
business_city_map = business_df[['business_id', 'city']]

# Filter to restaurants more robustly
restaurant_business = business_df[business_df['categories'].notna()]
restaurant_business = restaurant_business[restaurant_business['categories'].str.contains('Restaurant', case=False)]
restaurant_ids = set(restaurant_business['business_id'])

# Load and filter reviews in chunks
file_path = "data/raw/yelp_academic_dataset_review.json"
filtered_reviews = []

for chunk in pd.read_json(file_path, lines=True, chunksize=100_000):
    filtered = chunk[chunk['business_id'].isin(restaurant_ids)]
    filtered_reviews.append(filtered)

review_df = pd.concat(filtered_reviews)

# Join and format
ratings = review_df[['user_id', 'business_id', 'stars', 'date']]
ratings = ratings.merge(business_city_map, on='business_id', how='left')

# First split into train+val and prod
train_val = ratings.sample(frac=0.8, random_state=42)
prod = ratings.drop(train_val.index)

# Get unique users in train_val
train_val_users = set(train_val['user_id'])

# Split train_val into train and val
# First get all reviews from users that will be in train
train = train_val[train_val['user_id'].isin(train_val_users)].sample(frac=0.875, random_state=42)  # 0.875 * 0.8 = 0.7 of total
val = train_val.drop(train.index)

# Verify that all users in val are also in train
val_users = set(val['user_id'])
train_users = set(train['user_id'])
assert val_users.issubset(train_users), "Some users in validation set are not in training set"

logger.info(f"Train set size: {len(train)}")
logger.info(f"Validation set size: {len(val)}")
logger.info(f"Production set size: {len(prod)}")

# Save splits
train.to_csv("data/training/data.csv", index=False)
val.to_csv("data/validation/data.csv", index=False)
prod.to_csv("data/production/data.csv", index=False)
