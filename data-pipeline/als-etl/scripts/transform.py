import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Define input/output paths for Chameleon node
INPUT_DIR = "/staging/raw"
OUTPUT_DIR = "/staging/als"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load business data
business = pd.read_json(os.path.join(INPUT_DIR, 'yelp_academic_dataset_business.json'), lines=True)

# Extract only needed columns from business data
business_city_map = business[['business_id', 'city']]

# Filter only restaurants
restaurant_business = business[business['categories'].notna()]
restaurant_business = restaurant_business[restaurant_business['categories'].str.contains('Restaurant', case=False)]
restaurant_ids = set(restaurant_business['business_id'])

# Process reviews in chunks to handle large dataset
reviews_file = os.path.join(INPUT_DIR, 'yelp_academic_dataset_review.json')
chunks = pd.read_json(reviews_file, lines=True, chunksize=100_000)
filtered_reviews = []

print("Processing reviews in chunks...")
for chunk in chunks:
    # Filter restaurant reviews and keep only needed columns
    filtered = chunk[chunk['business_id'].isin(restaurant_ids)][['user_id', 'business_id', 'stars', 'date']]
    filtered_reviews.append(filtered)

# Combine all chunks
ratings = pd.concat(filtered_reviews)

# Add city information
ratings = ratings.merge(business_city_map, on='business_id', how='left')

# Ensure columns are in the correct order
ratings = ratings[['user_id', 'business_id', 'stars', 'date', 'city']]

# Convert date to datetime and sort
ratings['date'] = pd.to_datetime(ratings['date'])
ratings = ratings.sort_values(['user_id', 'date'])

# Split data into train/val/prod
def assign_splits(group):
    n = len(group)
    if n < 3:
        return pd.Series('train', index=group.index)
    labels = ['train'] * (n-2) + ['val', 'prod']
    return pd.Series(labels, index=group.index)

ratings['split'] = ratings.groupby('user_id', group_keys=False).apply(assign_splits)

# Create the splits
train = ratings[ratings.split == 'train'].drop('split', axis=1)
val = ratings[ratings.split == 'val'].drop('split', axis=1)
prod = ratings[ratings.split == 'prod'].drop('split', axis=1)

# Verify splits
assert set(val.user_id).issubset(train.user_id)
assert set(prod.user_id).issubset(train.user_id)

# Print statistics
print(f"Total ratings: {len(ratings)}")
print(f"Train: {len(train)}")
print(f"Val: {len(val)}")
print(f"Prod: {len(prod)}")

# Save the splits
train.to_csv(os.path.join(OUTPUT_DIR, "training_data.csv"), index=False)
val.to_csv(os.path.join(OUTPUT_DIR, "validation_data.csv"), index=False)
prod.to_csv(os.path.join(OUTPUT_DIR, "production_data.csv"), index=False)