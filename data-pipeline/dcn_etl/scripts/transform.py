import pandas as pd
import numpy as np
from pathlib import Path
import logging
import gc
import tempfile
import os
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Input paths
REVIEWS_PATH = "/staging/raw/yelp_academic_dataset_review.json"
BUSINESS_PATH = "/staging/raw/yelp_academic_dataset_business.json"
USER_VEC_PATH = "/staging/user_latent_vectors"
ITEM_VEC_PATH = "/staging/item_latent_vectors"

# Output paths
OUT_DIR = Path("/staging/dcn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUT_DIR / "train.parquet"
VAL_FILE = OUT_DIR / "validation.parquet"
PROD_FILE = OUT_DIR / "production.parquet"

# Temporary directory for intermediate files
TEMP_DIR = Path("/staging/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def load_business_data():
    logger.info("Loading business data and filtering restaurants...")
    business = pd.read_json(BUSINESS_PATH, lines=True)
    business_info = business[['business_id', 'categories', 'city']]
    del business
    gc.collect()
    
    # Filter restaurants
    restaurant_mask = (
        business_info['categories']
        .str.contains('Restaurant|Food', case=False, na=False)
    )
    restaurant_business = business_info[restaurant_mask][['business_id', 'city']].copy()
    del business_info
    gc.collect()
    
    logger.info(f"Found {len(restaurant_business)} restaurants")
    return restaurant_business

def load_latent_vectors():
    logger.info("Loading user latent vectors...")
    user_vec = pd.concat(
        [pd.read_csv(p) for p in sorted(Path(USER_VEC_PATH).glob("*.csv"))],
        ignore_index=True
    )
    user_vec = user_vec.rename(columns={"features": "user_latent_vector"})
    
    logger.info("Loading item latent vectors...")
    item_vec = pd.concat(
        [pd.read_csv(p) for p in sorted(Path(ITEM_VEC_PATH).glob("*.csv"))],
        ignore_index=True
    )
    item_vec = item_vec.rename(columns={"features": "item_latent_vector"})
    
    return user_vec, item_vec

def process_chunk(chunk, restaurant_business, user_vec, item_vec):
    # Extract relevant columns
    reviews = chunk[['review_id', 'user_id', 'business_id', 'stars', 'date']]
    reviews = reviews.rename(columns={'stars': 'rating'})
    
    # Filter restaurant reviews
    reviews = reviews[reviews['business_id'].isin(restaurant_business['business_id'])]
    
    # Merge all information
    merged = (reviews
             .merge(restaurant_business, on='business_id', how='inner')
             .merge(user_vec, on='user_id', how='left')
             .merge(item_vec, on='business_id', how='left'))
    
    return merged

def process_and_save_chunks(restaurant_business, user_vec, item_vec):
    """Process chunks and save them to temporary files"""
    logger.info("Processing reviews in chunks...")
    chunk_size = 100_000
    chunk_files = []
    
    for i, chunk in enumerate(pd.read_json(REVIEWS_PATH, lines=True, chunksize=chunk_size)):
        logger.info(f"Processing chunk {i+1}...")
        processed_chunk = process_chunk(chunk, restaurant_business, user_vec, item_vec)
        
        if len(processed_chunk) > 0:
            # Save chunk to temporary file
            temp_file = TEMP_DIR / f"chunk_{i}.parquet"
            processed_chunk.to_parquet(temp_file, index=False)
            chunk_files.append(temp_file)
        
        # Clean up
        del chunk, processed_chunk
        gc.collect()
    
    return chunk_files

def create_splits(chunk_files):
    """Create train/val/prod splits using a streaming approach"""
    logger.info("Creating data splits...")
    
    # Initialize split files
    train_file = TEMP_DIR / "train_temp.parquet"
    val_file = TEMP_DIR / "val_temp.parquet"
    prod_file = TEMP_DIR / "prod_temp.parquet"
    
    # Process each user's data separately
    user_counts = {}
    user_splits = {'train': [], 'val': [], 'prod': []}
    total_users = 0
    processed_users = 0
    processed_reviews = 0
    
    # First pass: count reviews per user
    logger.info("Counting reviews per user...")
    for i, chunk_file in enumerate(chunk_files):
        logger.info(f"Counting reviews in chunk {i+1}/{len(chunk_files)}...")
        df = pd.read_parquet(chunk_file)
        user_counts.update(df.groupby('user_id').size().to_dict())
    
    total_users = len(user_counts)
    eligible_users = sum(1 for count in user_counts.values() if count >= 3)
    logger.info(f"Found {total_users} total users, {eligible_users} users with 3+ reviews")
    
    # Second pass: assign reviews to splits
    logger.info("Assigning reviews to splits...")
    batch_size = 1000  # Number of users to process before saving
    save_counter = 0
    
    for chunk_idx, chunk_file in enumerate(chunk_files):
        logger.info(f"Processing chunk {chunk_idx+1}/{len(chunk_files)} for splits...")
        df = pd.read_parquet(chunk_file)
        df['date'] = pd.to_datetime(df['date'])
        chunk_users = df['user_id'].unique()
        logger.info(f"Found {len(chunk_users)} unique users in chunk")
        
        for user_id, user_data in df.groupby('user_id'):
            total_reviews = user_counts[user_id]
            if total_reviews < 3:
                continue
                
            processed_users += 1
            processed_reviews += len(user_data)
            
            user_data = user_data.sort_values('date')
            train_idx = int(total_reviews * 0.7)
            val_idx = int(total_reviews * 0.85)
            
            user_splits['train'].append(user_data.iloc[:train_idx])
            user_splits['val'].append(user_data.iloc[train_idx:val_idx])
            user_splits['prod'].append(user_data.iloc[val_idx:])
            
            # Periodically save and clear memory
            if len(user_splits['train']) >= batch_size:
                save_counter += 1
                logger.info(f"Saving batch {save_counter} ({processed_users}/{eligible_users} users, {processed_reviews:,} reviews processed)")
                
                for split_name, split_data in user_splits.items():
                    if split_data:
                        split_df = pd.concat(split_data, ignore_index=True)
                        if split_name == 'train':
                            split_df.to_parquet(train_file, index=False)
                        elif split_name == 'val':
                            split_df.to_parquet(val_file, index=False)
                        else:
                            split_df.to_parquet(prod_file, index=False)
                user_splits = {'train': [], 'val': [], 'prod': []}
                gc.collect()
                
        logger.info(f"Completed chunk {chunk_idx+1}, processed {processed_users:,}/{eligible_users:,} users ({(processed_users/eligible_users)*100:.1f}%)")
    
    # Save remaining data
    logger.info(f"Saving final batch ({processed_users}/{eligible_users} users processed)")
    for split_name, split_data in user_splits.items():
        if split_data:
            split_df = pd.concat(split_data, ignore_index=True)
            if split_name == 'train':
                split_df.to_parquet(TRAIN_FILE, index=False)
            elif split_name == 'val':
                split_df.to_parquet(VAL_FILE, index=False)
            else:
                split_df.to_parquet(PROD_FILE, index=False)
    
    logger.info("Split statistics:")
    logger.info(f"Total users processed: {processed_users:,}")
    logger.info(f"Total reviews processed: {processed_reviews:,}")
    
    # Clean up temporary files
    logger.info("Cleaning up temporary files...")
    try:
        for f in chunk_files:
            try:
                f.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")
        
        # Use shutil.rmtree instead of rmdir
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            logger.info("Successfully cleaned up temporary directory")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        # Continue execution even if cleanup fails
        pass

def main():
    try:
        # Load business data
        restaurant_business = load_business_data()
        
        # Load latent vectors
        user_vec, item_vec = load_latent_vectors()
        
        # Process chunks and save to temporary files
        chunk_files = process_and_save_chunks(restaurant_business, user_vec, item_vec)
        
        # Create splits from temporary files
        create_splits(chunk_files)
        
        logger.info("Transform completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during transformation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()


