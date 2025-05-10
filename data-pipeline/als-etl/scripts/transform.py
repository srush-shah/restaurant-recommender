import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
from collections import Counter
import multiprocessing as mp
from functools import partial
import sys
import gc
import logging
import traceback
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/staging/transform.log')
    ]
)

def log_memory_usage():
    """Log current memory usage"""
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024  # in MB
    logging.info(f"Current memory usage: {mem:.2f} MB")

# Define input/output paths for Chameleon node
INPUT_DIR = "/staging/raw"
OUTPUT_DIR = "/staging/als"

# Set pandas options for better performance
pd.options.mode.chained_assignment = None
chunk_size = int(os.environ.get('PANDAS_CHUNKSIZE', 100000))

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_categories(categories_str):
    if pd.isna(categories_str):
        return []
    return [cat.strip() for cat in categories_str.split(',')]

def create_category_encoding(business_df, min_category_freq=10):
    """Create category encoding from business data"""
    try:
        logging.info("Creating category encoding...")
        all_categories = []
        for cats in business_df['categories'].dropna():
            all_categories.extend(process_categories(cats))
        
        category_counts = Counter(all_categories)
        
        valid_categories = {cat for cat, count in category_counts.items() 
                          if count >= min_category_freq and 'Restaurant' not in cat}
        
        category_to_idx = {cat: idx for idx, cat in enumerate(sorted(valid_categories))}
        logging.info(f"Found {len(category_to_idx)} valid categories")
        return category_to_idx
    except Exception as e:
        logging.error(f"Error in create_category_encoding: {e}")
        logging.error(traceback.format_exc())
        raise

def encode_business_categories(business_df, category_to_idx):
    """Create binary category vectors for each business"""
    try:
        logging.info("Encoding business categories...")
        n_categories = len(category_to_idx)
        business_categories = {}
        
        total = len(business_df)
        for idx, row in business_df.iterrows():
            if idx % 1000 == 0:
                logging.info(f"Encoded {idx}/{total} businesses...")
            
            if pd.isna(row['categories']):
                business_categories[row['business_id']] = np.zeros(n_categories)
                continue
                
            categories = process_categories(row['categories'])
            category_vector = np.zeros(n_categories)
            
            for cat in categories:
                if cat in category_to_idx:
                    category_vector[category_to_idx[cat]] = 1
                    
            business_categories[row['business_id']] = category_vector
        
        logging.info(f"Encoded categories for {len(business_categories)} businesses")
        return business_categories
    except Exception as e:
        logging.error(f"Error in encode_business_categories: {e}")
        logging.error(traceback.format_exc())
        raise

def process_chunk(chunk_data):
    chunk, restaurant_ids = chunk_data
    try:
        filtered = chunk[chunk['business_id'].isin(restaurant_ids)][['user_id', 'business_id', 'stars', 'date']]
        return filtered
    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
        logging.error(traceback.format_exc())
        return pd.DataFrame()

def save_dataframe(df, filename, output_dir):
    """Save dataframe in chunks to avoid memory issues"""
    try:
        logging.info(f"Saving {filename}...")
        chunk_size = 100000  # 100K rows per chunk
        for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
            chunk = df.iloc[chunk_start:chunk_start + chunk_size]
            mode = 'w' if i == 0 else 'a'
            header = i == 0
            chunk.to_csv(os.path.join(output_dir, filename), 
                        mode=mode, header=header, index=False)
            del chunk
            gc.collect()
            if i % 10 == 0:
                logging.info(f"Saved chunk {i} of {filename}")
    except Exception as e:
        logging.error(f"Error in save_dataframe: {e}")
        logging.error(traceback.format_exc())
        raise

def main():
    try:
        start_time = datetime.now()
        logging.info("Starting transformation process...")
        log_memory_usage()

        # Load business data
        logging.info("Loading business data...")
        business = pd.read_json(os.path.join(INPUT_DIR, 'yelp_academic_dataset_business.json'), lines=True)
        business_city_map = business[['business_id', 'city']]
        log_memory_usage()

        # Filter restaurants
        logging.info("Filtering restaurants...")
        restaurant_business = business[business['categories'].notna()]
        restaurant_business = restaurant_business[
            restaurant_business['categories'].str.contains('(Food|Restaurant)', 
                                                         case=False, 
                                                         regex=True)]
        restaurant_ids = set(restaurant_business['business_id'])
        logging.info(f"Found {len(restaurant_ids)} restaurants")
        
        # Free up memory
        del business
        gc.collect()
        log_memory_usage()

        # Process categories
        category_to_idx = create_category_encoding(restaurant_business)
        business_categories = encode_business_categories(restaurant_business, category_to_idx)
        log_memory_usage()

        # Convert to DataFrame
        logging.info("Converting categories to DataFrame...")
        category_cols = [f'category_{i}' for i in range(len(category_to_idx))]
        business_categories_df = pd.DataFrame.from_dict(business_categories, orient='index', 
                                                      columns=category_cols)
        business_categories_df.index.name = 'business_id'
        business_categories_df = business_categories_df.reset_index()
        log_memory_usage()

        # Save metadata
        logging.info("Saving category metadata...")
        category_metadata = {
            'category_to_idx': category_to_idx,
            'n_categories': len(category_to_idx),
            'category_columns': category_cols
        }
        with open(os.path.join(OUTPUT_DIR, 'category_metadata.json'), 'w') as f:
            json.dump(category_metadata, f)

        # Process reviews
        logging.info("Processing reviews...")
        reviews_file = os.path.join(INPUT_DIR, 'yelp_academic_dataset_review.json')
        n_workers = min(mp.cpu_count(), 8)
        logging.info(f"Using {n_workers} workers for parallel processing...")

        # Create temporary directory for chunks
        temp_dir = os.path.join(OUTPUT_DIR, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Process reviews in chunks
        chunk_files = []
        total_processed = 0
        chunks = pd.read_json(reviews_file, lines=True, chunksize=chunk_size)

        for chunk_num, chunk in enumerate(chunks):
            try:
                logging.info(f"Processing chunk {chunk_num}...")
                filtered = process_chunk((chunk, restaurant_ids))
                
                if not filtered.empty:
                    chunk_file = os.path.join(temp_dir, f'chunk_{chunk_num}.csv')
                    filtered.to_csv(chunk_file, index=False)
                    chunk_files.append(chunk_file)
                    total_processed += len(filtered)
                    
                    logging.info(f"Processed {total_processed:,} reviews total")
                    log_memory_usage()
                
                del filtered, chunk
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error processing chunk {chunk_num}: {e}")
                logging.error(traceback.format_exc())
                continue

        # Process saved chunks and create final datasets
        logging.info("Creating final datasets...")
        train_file = os.path.join(OUTPUT_DIR, "training_data.csv")
        val_file = os.path.join(OUTPUT_DIR, "validation_data.csv")
        prod_file = os.path.join(OUTPUT_DIR, "production_data.csv")

        # Initialize files with headers
        for file in [train_file, val_file, prod_file]:
            pd.DataFrame(columns=['user_id', 'business_id', 'stars', 'date', 'city'] + category_cols).to_csv(file, index=False)

        # Process each saved chunk
        for i, chunk_file in enumerate(chunk_files):
            try:
                logging.info(f"Processing saved chunk {i+1}/{len(chunk_files)}...")
                chunk = pd.read_csv(chunk_file)
                chunk['date'] = pd.to_datetime(chunk['date'])
                
                # Add city and category information
                chunk = chunk.merge(business_city_map, on='business_id', how='left')
                chunk = chunk.merge(business_categories_df, on='business_id', how='left')

                # Split the chunk
                chunk['split'] = chunk.groupby('user_id')['date'].rank(pct=True).apply(
                    lambda x: 'train' if x <= 0.8 else ('val' if x <= 0.9 else 'prod'))

                # Save to respective files
                for split, file in [('train', train_file), ('val', val_file), ('prod', prod_file)]:
                    split_chunk = chunk[chunk['split'] == split].drop('split', axis=1)
                    split_chunk.to_csv(file, mode='a', header=False, index=False)

                # Cleanup
                os.remove(chunk_file)
                del chunk
                gc.collect()
                log_memory_usage()

            except Exception as e:
                logging.error(f"Error processing saved chunk {i}: {e}")
                logging.error(traceback.format_exc())
                continue

        # Clean up
        logging.info("Cleaning up temporary files...")
        os.rmdir(temp_dir)

        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Transform completed successfully! Duration: {duration}")

    except Exception as e:
        logging.error(f"Fatal error during transformation: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()