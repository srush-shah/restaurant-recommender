import pandas as pd
import numpy as np
import os
import json
from collections import Counter
import multiprocessing as mp
import sys
import gc
import logging
import traceback
from datetime import datetime
import random

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

# Splitting parameters (can be changed as required)
MIN_INTERACTIONS_PER_USER = 3
MIN_INTERACTIONS_PER_ITEM = 3
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
# TEST_RATIO is implicitly 1.0 - TRAIN_RATIO - VALIDATION_RATIO (i.e., 0.2)

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

def apply_k_core_filtering(df, min_user=3, min_item=3):
    """Apply K-core filtering iteratively until convergence"""
    try:
        logging.info(f"Starting K-core filtering with min_user={min_user}, min_item={min_item}")
        initial_count = len(df)
        logging.info(f"Initial dataframe size: {initial_count} rows")
        
        # Iterative filtering
        converged = False
        iteration = 1
        current_df = df.copy()
        
        while not converged:
            logging.info(f"K-core iteration {iteration}: Starting with {len(current_df)} records")
            
            # Filter by user count
            user_counts = current_df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_user].index
            df_after_user_filter = current_df[current_df['user_id'].isin(valid_users)]
            
            # Filter by item count
            item_counts = df_after_user_filter['business_id'].value_counts()
            valid_items = item_counts[item_counts >= min_item].index
            new_df = df_after_user_filter[df_after_user_filter['business_id'].isin(valid_items)]
            
            new_count = len(new_df)
            logging.info(f"K-core iteration {iteration}: Ended with {new_count} records")
            
            # Check for convergence
            if new_count == len(current_df):
                converged = True
                logging.info("K-core filtering converged")
            
            current_df = new_df
            iteration += 1
            
            # Safety check for max iterations
            if iteration > 10:
                logging.warning("K-core reached max iterations (10). Breaking loop.")
                break
        
        logging.info(f"K-core filtering complete. Records after filtering: {len(current_df)} (removed {initial_count - len(current_df)} records)")
        return current_df
    
    except Exception as e:
        logging.error(f"Error in k-core filtering: {e}")
        logging.error(traceback.format_exc())
        raise

def stratified_train_test_split(df):
    """Split data ensuring each user has items in train, validation, and test sets"""
    try:
        logging.info("Performing stratified train-test split")
        
        # Ensure each user has a minimum number of interactions
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= MIN_INTERACTIONS_PER_USER].index
        df_valid = df[df['user_id'].isin(valid_users)]
        
        # For each user, split their interactions
        train_data = []
        validation_data = []
        test_data = []
        
        # Set a random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # For each user, split interactions preserving time order where possible
        user_groups = df_valid.groupby('user_id')
        total_users = len(user_groups)
        
        for i, (user_id, user_df) in enumerate(user_groups):
            if i % 1000 == 0:
                logging.info(f"Splitting data for user {i}/{total_users}")
            
            # Sort by date if available to respect chronological order
            if 'date' in user_df.columns:
                user_df = user_df.sort_values('date')
            
            # Calculate split points
            n_interactions = len(user_df)
            train_size = int(n_interactions * TRAIN_RATIO)
            val_size = int(n_interactions * VALIDATION_RATIO)
            
            # Ensure minimum 1 item in each split if possible
            if n_interactions >= 3:
                train_size = max(train_size, 1)
                val_size = max(val_size, 1)
                test_size = n_interactions - train_size - val_size
                test_size = max(test_size, 1)
                
                # Rebalance if necessary
                if train_size + val_size + test_size > n_interactions:
                    if val_size > 1:
                        val_size -= 1
                    elif train_size > 1:
                        train_size -= 1
            
            # Split the data
            train_data.append(user_df.iloc[:train_size])
            validation_data.append(user_df.iloc[train_size:train_size+val_size])
            test_data.append(user_df.iloc[train_size+val_size:])
        
        # Combine all user splits
        train_df = pd.concat(train_data, ignore_index=True)
        validation_df = pd.concat(validation_data, ignore_index=True)
        test_df = pd.concat(test_data, ignore_index=True)
        
        logging.info(f"Split complete: Train={len(train_df)}, Validation={len(validation_df)}, Test={len(test_df)}")
        
        # Ensure warm-start for validation and test sets (items must exist in train)
        train_items = set(train_df['business_id'].unique())
        validation_df = validation_df[validation_df['business_id'].isin(train_items)]
        test_df = test_df[test_df['business_id'].isin(train_items)]
        
        logging.info(f"After warm-start filtering: Train={len(train_df)}, Validation={len(validation_df)}, Test={len(test_df)}")
        
        return train_df, validation_df, test_df
    
    except Exception as e:
        logging.error(f"Error in stratified train test split: {e}")
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

        # Combine all chunks into a single dataframe
        logging.info("Combining all processed chunks...")
        combined_df = pd.DataFrame()
        for i, chunk_file in enumerate(chunk_files):
            try:
                logging.info(f"Reading chunk file {i+1}/{len(chunk_files)}...")
                chunk = pd.read_csv(chunk_file)
                chunk['date'] = pd.to_datetime(chunk['date'])
                combined_df = pd.concat([combined_df, chunk], ignore_index=True)
                
                # Cleanup individual chunk files early to save space
                os.remove(chunk_file)
                del chunk
                gc.collect()
                
            except Exception as e:
                logging.error(f"Error reading chunk file {i}: {e}")
                logging.error(traceback.format_exc())
                continue
                
        logging.info(f"Combined dataframe size: {len(combined_df)} rows")
        log_memory_usage()
        
        # Add city and category information
        logging.info("Adding business metadata (city and categories)...")
        combined_df = combined_df.merge(business_city_map, on='business_id', how='left')
        combined_df = combined_df.merge(business_categories_df, on='business_id', how='left')
        logging.info(f"Dataframe size after adding metadata: {len(combined_df)} rows")
        log_memory_usage()
        
        # Apply K-core filtering
        filtered_df = apply_k_core_filtering(
            combined_df, 
            min_user=MIN_INTERACTIONS_PER_USER, 
            min_item=MIN_INTERACTIONS_PER_ITEM
        )
        
        # Free memory from the original combined dataframe
        del combined_df
        gc.collect()
        log_memory_usage()
        
        # Apply stratified train-test split
        train_df, validation_df, test_df = stratified_train_test_split(filtered_df)
        
        # Free memory from the filtered dataframe
        del filtered_df
        gc.collect()
        log_memory_usage()
        
        # Save the train, validation, and test splits
        logging.info("Saving final datasets...")
        train_file = os.path.join(OUTPUT_DIR, "training_data.csv")
        val_file = os.path.join(OUTPUT_DIR, "validation_data.csv")
        prod_file = os.path.join(OUTPUT_DIR, "production_data.csv")  # 'prod' is our test set
        
        save_dataframe(train_df, "training_data.csv", OUTPUT_DIR)
        save_dataframe(validation_df, "validation_data.csv", OUTPUT_DIR)
        save_dataframe(test_df, "production_data.csv", OUTPUT_DIR)
        
        # Clean up remaining temporary files
        try:
            os.rmdir(temp_dir)
        except:
            logging.warning(f"Could not remove temp directory {temp_dir}. It may not be empty.")

        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Transform completed successfully! Duration: {duration}")
        logging.info(f"Final dataset sizes: Train={len(train_df)}, Validation={len(validation_df)}, Test={len(test_df)}")

    except Exception as e:
        logging.error(f"Fatal error during transformation: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()