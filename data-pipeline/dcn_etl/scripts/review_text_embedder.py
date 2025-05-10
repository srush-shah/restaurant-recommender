import torch
import pandas as pd
import json
from transformers import BertTokenizer, BertModel
from tqdm.notebook import tqdm # For progress bars, especially useful for long operations
import gc # Garbage collector

print(f"PyTorch version: {torch.__version__}")
print(f"Pandas version: {pd.__version__}")

# Check for GPU availability and set the device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch is using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("PyTorch is using CPU. Check your Docker GPU setup if this is unexpected.")

# Load the tokenizer for bert-base-uncased
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("BERT tokenizer loaded.")

# Load the pre-trained bert-base-uncased model
model = BertModel.from_pretrained('bert-base-uncased')
print("BERT model loaded.")

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model) # Wrap the model

# Move the model to the selected device (GPU)
model.to(device)
print(f"Model moved to {device}.")

# Set the model to evaluation mode (important if you're not fine-tuning)
model.eval()
print("Model set to evaluation mode.")


json_file_path = './data/raw/yelp_academic_dataset_review.json'

# Define the field names from your JSON objects
# (Based on your screenshot)
review_id_field = 'review_id'
user_id_field = 'user_id'
business_id_field = 'business_id'
stars_field = 'stars' # We'll extract it but might not use it directly for embedding
date_field = 'date'
text_field = 'text'

# Set to small number to test Or None for all records
MAX_RECORDS_TO_LOAD = 1_000_000

print(f"JSON file path: {json_file_path}")
if MAX_RECORDS_TO_LOAD:
    print(f"Will attempt to load a maximum of {MAX_RECORDS_TO_LOAD} records for initial processing.")
else:
    print("Will attempt to load all records from the file.")

# Initialize lists to store extracted data
review_ids_list = []
user_ids_list = []
business_ids_list = []
dates_list = []
texts_to_process_list = []
stars_list = [] # Storing stars as well

print("Starting data extraction from JSON Lines file...")
lines_processed = 0
records_loaded = 0

try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading lines", unit="line"):
            lines_processed += 1
            try:
                record = json.loads(line) # Parse each line as a JSON object
                
                # Extract data if fields exist and text is a string
                text_content = record.get(text_field)
                if (record.get(review_id_field) and
                    record.get(user_id_field) and
                    record.get(business_id_field) and
                    record.get(date_field) and
                    text_content and isinstance(text_content, str)):
                    
                    review_ids_list.append(record[review_id_field])
                    user_ids_list.append(record[user_id_field])
                    business_ids_list.append(record[business_id_field])
                    dates_list.append(record[date_field])
                    texts_to_process_list.append(text_content)
                    stars_list.append(record.get(stars_field)) # .get() is safer for optional fields
                    
                    records_loaded += 1
                else:
                    if lines_processed <= 100: # Print warnings only for the first few lines to avoid flooding output
                        print(f"Warning: Skipping record on line {lines_processed} due to missing fields or non-string text.")
                
            except json.JSONDecodeError:
                if lines_processed <= 100:
                    print(f"Warning: JSONDecodeError on line {lines_processed}. Skipping.")
            except Exception as e:
                if lines_processed <= 100:
                    print(f"Warning: An unexpected error occurred processing line {lines_processed}: {e}. Skipping.")

            if MAX_RECORDS_TO_LOAD and records_loaded >= MAX_RECORDS_TO_LOAD:
                print(f"\nReached MAX_RECORDS_TO_LOAD limit of {MAX_RECORDS_TO_LOAD}. Stopping data loading.")
                break
                
except FileNotFoundError:
    print(f"ERROR: File not found at {json_file_path}. Please check the path.")
except Exception as e:
    print(f"ERROR: An unexpected error occurred opening or reading the file: {e}")

print(f"\nFinished data extraction.")
print(f"Total lines processed from file: {lines_processed}")
print(f"Total valid records loaded: {records_loaded}")

if records_loaded > 0:
    print("\nSample of loaded data (first 3 records):")
    for i in range(min(3, records_loaded)):
        print(f"  Review ID: {review_ids_list[i]}, User ID: {user_ids_list[i]}, Business ID: {business_ids_list[i]}, Date: {dates_list[i]}, Stars: {stars_list[i]}, Text: {texts_to_process_list[i][:60]}...")
else:
    print("No records were loaded. Please check your file, path, and MAX_RECORDS_TO_LOAD setting.")


if records_loaded > 0:
    source_data_df = pd.DataFrame({
        review_id_field: review_ids_list,
        user_id_field: user_ids_list,
        business_id_field: business_ids_list,
        date_field: dates_list,
        stars_field: stars_list,
        text_field: texts_to_process_list
    })
    
    print("\nDataFrame created from loaded data:")
    print(f"Shape of DataFrame: {source_data_df.shape}")
    print("\nFirst 5 rows:")
    print(source_data_df.head())
    print("\nInfo:")
    source_data_df.info(memory_usage='deep') # memory_usage='deep' gives a more accurate size
    
    # Clear original lists to free up memory if DataFrame is successfully created
    del review_ids_list, user_ids_list, business_ids_list, dates_list, texts_to_process_list, stars_list
    gc.collect() # Invoke garbage collector
    print("\nOriginal lists cleared from memory.")
    
else:
    print("\nNo data loaded, DataFrame not created.")
    source_data_df = pd.DataFrame() # Create an empty DataFrame

def get_bert_embeddings_batched(texts, model, tokenizer, device, batch_size=16, desc="Embedding"):
    """
    Generates BERT embeddings for a list of texts in batches.
    Uses pooler_output as the embedding for each text.
    """
    model.eval()
    all_pooler_embeddings = []
    
    # Wrap range with tqdm for a progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch_texts = texts[i:i + batch_size]
        
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512, # BERT's max sequence length
            return_tensors='pt'
        )
        
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
        
        with torch.no_grad():
            outputs = model(**encoded_input)
        
        batch_pooler_embeddings = outputs.pooler_output
        all_pooler_embeddings.append(batch_pooler_embeddings.cpu()) # Move to CPU to save GPU RAM

    return torch.cat(all_pooler_embeddings, dim=0)

print("BERT embedding function defined.")

EMBEDDING_BATCH_SIZE = 768

if not source_data_df.empty:
    print(f"Preparing to generate embeddings for {len(source_data_df)} texts with batch size {EMBEDDING_BATCH_SIZE}...")
    
    # Get the list of texts from the DataFrame
    texts_for_embedding = source_data_df[text_field].tolist()
    
    # Generate embeddings
    # This is the most time-consuming step for large datasets.
    bert_embeddings_tensor = get_bert_embeddings_batched(
        texts_for_embedding, 
        model, 
        tokenizer, 
        device, 
        batch_size=EMBEDDING_BATCH_SIZE,
        desc="Generating Embeddings"
    )
    
    print(f"\nEmbedding generation complete.")
    print(f"Shape of BERT embeddings tensor: {bert_embeddings_tensor.shape}") # Should be (num_texts, 768)

    # Optional: Clear the list of texts if memory is very tight, once embeddings are generated
    # del texts_for_embedding
    # gc.collect()

else:
    print("Source DataFrame is empty. Skipping embedding generation.")
    bert_embeddings_tensor = None

if bert_embeddings_tensor is not None and not source_data_df.empty:
    if len(source_data_df) == bert_embeddings_tensor.shape[0]:
        print("Adding embeddings to the DataFrame...")
        
        # Convert tensor to a list of lists/arrays for easier DataFrame storage
        embeddings_list = [emb.tolist() for emb in bert_embeddings_tensor]
        
        # Assign as a new column
        # Make a copy to avoid SettingWithCopyWarning if source_data_df is a slice from a larger df (not the case here but good practice)
        final_df = source_data_df.copy()
        final_df['embedding'] = embeddings_list
        
        print("\nFinal DataFrame with embeddings (first 5 rows):")
        print(final_df.head())
        print(f"\nShape of final DataFrame: {final_df.shape}")
        print("\nInfo for final DataFrame:")
        final_df.info(memory_usage='deep')

        # We can now remove the bert_embeddings_tensor and embeddings_list if RAM is a concern
        # del bert_embeddings_tensor, embeddings_list
        # gc.collect()
    else:
        print("ERROR: Mismatch in the number of records and embeddings. Cannot combine.")
        final_df = pd.DataFrame() # Assign empty df
else:
    print("No embeddings generated or source DataFrame is empty. Final DataFrame not created.")
    final_df = pd.DataFrame() # Assign empty df

if not final_df.empty:
    # Define the columns you want in your final CSV output
    columns_to_save = [
        review_id_field,
        user_id_field,
        business_id_field,
        date_field,
        stars_field,
        'embedding'
    ]
    
    # Check if all desired columns exist in final_df
    missing_cols = [col for col in columns_to_save if col not in final_df.columns]
    if missing_cols:
        print(f"ERROR: The following requested columns are missing from final_df: {missing_cols}")
        print(f"Available columns in final_df are: {final_df.columns.tolist()}")
        print("Cannot save CSV with specified headers. Please check column names and previous cells.")
    else:
        # Create a DataFrame with only the specified columns in the desired order
        df_for_csv = final_df[columns_to_save]

        # Define output file path for CSV
        # MAX_RECORDS_TO_LOAD is from your Cell 4 configuration
        csv_file_name = f"processed_reviews_embeddings_{MAX_RECORDS_TO_LOAD if MAX_RECORDS_TO_LOAD else 'all'}.csv"
        output_csv_path = f"/home/jovyan/work/{csv_file_name}" # Ensure /home/jovyan/work/ is writable
        
        print(f"\nSaving selected columns to CSV: {output_csv_path}")
        try:
            # Save to CSV, without the DataFrame index
            df_for_csv.to_csv(output_csv_path, index=False)
            print(f"Successfully saved data to CSV: {output_csv_path}")
            print("\nFirst 5 rows of the CSV file (as a DataFrame) would look like:")
            print(df_for_csv.head())
        except Exception as e:
            print(f"ERROR saving to CSV: {e}")
else:
    print("Final DataFrame is empty. Nothing to save.")