import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import time
import mlflow
import mlflow.pytorch
import subprocess

mlflow.set_experiment("restaurant-rating-mlp")

USER_VECTOR_CSV_STRING_ID_COL = 'user_id'
USER_VECTOR_CSV_VECTOR_STRING_COL = 'features'

ITEM_VECTOR_CSV_STRING_ID_COL = 'business_id'
ITEM_VECTOR_CSV_VECTOR_STRING_COL = 'features'

# Load the data
try:
    df_main = pd.read_csv('./data/als/training_data.csv')

    # Load only the necessary columns from the vector CSVs
    user_vectors_df_raw = pd.read_csv(
        'user-latent-vectors.csv',
        usecols=[USER_VECTOR_CSV_STRING_ID_COL, USER_VECTOR_CSV_VECTOR_STRING_COL],
        dtype={USER_VECTOR_CSV_VECTOR_STRING_COL: str}
    )
    item_vectors_df_raw = pd.read_csv(
        'item-latent-vectors.csv',
        usecols=[ITEM_VECTOR_CSV_STRING_ID_COL, ITEM_VECTOR_CSV_VECTOR_STRING_COL],
        dtype={ITEM_VECTOR_CSV_VECTOR_STRING_COL: str}
    )
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
    raise
except ValueError as e: # Handles errors from usecols if a specified column doesn't exist
    print(f"ValueError during CSV loading. Check if your specified ID and vector string columns exist in the CSVs: {e}")
    raise

def parse_comma_separated_float_string(vector_string):
    """
    Parses a string of comma-separated float values.
    Example input: "-0.375,0.948,..."
    Returns a list of floats, or an empty list if parsing fails or input is invalid.
    """
    if pd.isna(vector_string) or not isinstance(vector_string, str) or not vector_string.strip():
        return []
    try:
        return [float(x.strip()) for x in vector_string.split(',')]
    except ValueError:
        return []

def parse_and_expand_vector_column(df_raw, id_col_name_for_merge, single_vec_str_col_name, vec_prefix):
    if df_raw.empty or single_vec_str_col_name not in df_raw.columns:
        print(f"Warning: DataFrame for '{vec_prefix}' is empty or vector string column '{single_vec_str_col_name}' not found.")
        return pd.DataFrame(columns=[id_col_name_for_merge])

    # Apply the new parsing function
    parsed_vectors = df_raw[single_vec_str_col_name].apply(parse_comma_separated_float_string)

    # Create a DataFrame from the list of lists (each sublist is a vector)
    vector_components_df = pd.DataFrame(parsed_vectors.tolist(), index=df_raw.index)

    if vector_components_df.empty and not parsed_vectors.empty:
        print(f"Warning: All vectors for '{vec_prefix}' were empty or unparseable. No component columns created from data.")
    if not vector_components_df.empty:
        vector_components_df.columns = [f"{vec_prefix}{i}" for i in range(vector_components_df.shape[1])]
        vector_components_df = vector_components_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    else:
        print(f"Warning: No vector component columns created for '{vec_prefix}' because parsed data led to empty DataFrame.")

    expanded_df = pd.concat([df_raw[[id_col_name_for_merge]], vector_components_df], axis=1)
    return expanded_df

# Apply the parsing and expansion
print(f"\nProcessing user vectors. ID col for merge: '{USER_VECTOR_CSV_STRING_ID_COL}', Vector string col: '{USER_VECTOR_CSV_VECTOR_STRING_COL}'")
user_vectors_expanded_df = parse_and_expand_vector_column(
    user_vectors_df_raw,
    USER_VECTOR_CSV_STRING_ID_COL,
    USER_VECTOR_CSV_VECTOR_STRING_COL,
    "user_vec_"
)

print(f"\nProcessing item vectors. ID col for merge: '{ITEM_VECTOR_CSV_STRING_ID_COL}', Vector string col: '{ITEM_VECTOR_CSV_VECTOR_STRING_COL}'")
item_vectors_expanded_df = parse_and_expand_vector_column(
    item_vectors_df_raw,
    ITEM_VECTOR_CSV_STRING_ID_COL,
    ITEM_VECTOR_CSV_VECTOR_STRING_COL,
    "item_vec_"
)
MAIN_DF_USER_ID_COL = 'user_id'
MAIN_DF_ITEM_ID_COL = 'business_id'

df_merged = pd.merge(df_main, user_vectors_expanded_df,
                     left_on=MAIN_DF_USER_ID_COL, right_on=USER_VECTOR_CSV_STRING_ID_COL,
                     how='left')
if MAIN_DF_USER_ID_COL != USER_VECTOR_CSV_STRING_ID_COL and USER_VECTOR_CSV_STRING_ID_COL in df_merged.columns:
    df_merged.drop(columns=[USER_VECTOR_CSV_STRING_ID_COL], inplace=True)

df_merged = pd.merge(df_merged, item_vectors_expanded_df,
                     left_on=MAIN_DF_ITEM_ID_COL, right_on=ITEM_VECTOR_CSV_STRING_ID_COL,
                     how='left')
if MAIN_DF_ITEM_ID_COL != ITEM_VECTOR_CSV_STRING_ID_COL and ITEM_VECTOR_CSV_STRING_ID_COL in df_merged.columns:
    df_merged.drop(columns=[ITEM_VECTOR_CSV_STRING_ID_COL], inplace=True)
df_merged = df_merged.fillna(0.0)

X_model_input = df_merged.select_dtypes(include=np.number)

# First, split into training and a temporary set (validation + test)
df_train, df_temp = train_test_split(X_model_input, test_size=0.3, random_state=42)

# Then, split the temporary set into validation and test
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

print(f"Training set shape: {df_train.shape}")
print(f"Validation set shape: {df_val.shape}")
print(f"Test set shape: {df_test.shape}")

all_cols = df_train.columns.tolist()

col_exclude = 'stars'

model_feature_columns = [col for col in all_cols if col not in col_exclude]

# Training set
X_train = df_train[model_feature_columns]
y_train = df_train['stars'] # Use the scaled target

# Validation set
X_val = df_val[model_feature_columns]
y_val = df_val['stars']   # Use the scaled target

# Test set
X_test = df_test[model_feature_columns]
y_test = df_test['stars']   # Use the scaled target

print("\nShapes after separating X and y:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Verify no NaNs in the final X sets (especially important after scaling/transforms)
print(f"\nNaNs in X_train: {X_train.isnull().sum().sum()}")
print(f"NaNs in X_val: {X_val.isnull().sum().sum()}")
print(f"NaNs in X_test: {X_test.isnull().sum().sum()}")

# Convert to PyTorch Tensors
print("Converting data to tensors...")
# Training set
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

# Validation set
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

# Test set
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
print("Data tensor conversion complete.")

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 4096
num_data_workers = 8

print(f"Creating DataLoaders with batch_size={batch_size} and num_workers={num_data_workers}...")
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True, # Shuffle training data
    num_workers=num_data_workers,
    pin_memory=True
)
val_loader = DataLoader( # Validation loader
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False, # No need to shuffle validation or test data
    num_workers=num_data_workers,
    pin_memory=True
)
test_loader = DataLoader( # Test loader
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_data_workers,
    pin_memory=True
)
print("DataLoaders created.")

class SimpleMLP(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim_1, hidden_dim_2, output_size=1): 
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_feature_dim, hidden_dim_1)
        self.relu1 = nn.ReLU()                        
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.relu2 = nn.ReLU()
        self.output_logits = nn.Linear(hidden_dim_2, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        logits = self.output_logits(x)
        
        # Scale sigmoid output to be between 0 and 5
        rating = 5.0 * torch.sigmoid(logits)
        return rating

input_dim = X_train_tensor.shape[1]
hidden_layer_1_size = 256
hidden_layer_2_size = 128
output_dim = 1

print("Instantiating model with two hidden layers...")
model = SimpleMLP(input_feature_dim=input_dim,
                  hidden_dim_1=hidden_layer_1_size,  
                  hidden_dim_2=hidden_layer_2_size, 
                  output_size=output_dim)

criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set up device
if torch.cuda.is_available():
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available! Using {num_gpus} GPU(s):")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Wrap model with DataParallel if multiple GPUs are available
    if num_gpus > 1:
        print(f"Wrapping model with nn.DataParallel for {num_gpus} GPUs.")
        model = nn.DataParallel(model) # This is the key change for DataParallel
else:
    device = torch.device("cpu")
    print("CUDA not available, falling back to CPU.")

model.to(device)

print(f"Model moved to device: {device}")
print(f"Input feature dimension: {input_dim}")
print(f"Model architecture: {model}")

try: 
    mlflow.end_run() # end pre-existing run, if there was one
except:
    pass
finally:
    mlflow.start_run(log_system_metrics=True) # Start MLFlow run
    # automatically log GPU and CPU metrics
    # Note: to automatically log AMD GPU metrics, you need to have installed pyrsmi
    # Note: to automatically log NVIDIA GPU metrics, you need to have installed pynvml

# Let's get the output of rocm-info or nvidia-smi as a string...
gpu_info = next(
    (subprocess.run(cmd, capture_output=True, text=True).stdout for cmd in ["nvidia-smi", "rocm-smi"] if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode == 0),
    "No GPU found."
)
# ... and send it to MLFlow as a text file
mlflow.log_text(gpu_info, "gpu-info.txt")

num_epochs = 20 # Adjust as needed

print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs): # Standard Python range: 0 to num_epochs-1
    model.train() # Set model to training mode
    total_train_loss = 0
    epoch_start_time = time.time()

    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        predictions = model(features)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    epoch_end_time = time.time()
    # Use epoch + 1 for 1-based indexing in print statements
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss (MSE): {avg_train_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s')

    # Validation phase (evaluate on validation set after each epoch)
    model.eval() # Set model to evaluation mode
    total_val_loss_mse = 0
    correct_val_predictions = 0
    total_val_samples = 0
    with torch.no_grad():
        for features, labels in val_loader: # Use val_loader here
            features, labels = features.to(device), labels.to(device)
            predictions = model(features) # Model outputs continuous values (0-5)
            val_loss = criterion(predictions, labels)
            total_val_loss_mse += val_loss.item()

            # Accuracy calculation for validation set
            rounded_predictions = torch.round(predictions)
            correct_val_predictions += (rounded_predictions == labels).sum().item()
            total_val_samples += labels.size(0)

    avg_val_loss_mse = total_val_loss_mse / len(val_loader)
    avg_val_loss_rmse = np.sqrt(avg_val_loss_mse)
    val_accuracy = (correct_val_predictions / total_val_samples) * 100 if total_val_samples > 0 else 0
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss (MSE): {avg_val_loss_mse:.4f}, Validation RMSE: {avg_val_loss_rmse:.4f}, Validation Accuracy (Rounded): {val_accuracy:.2f}%')
    print("-" * 50) # Increased separator length for clarity

print("Training complete.")

# --- Final Evaluation on the Test Set ---
# This should be done only once, after all training and validation is complete.
print("\nEvaluating on the Test Set...")
model.eval() # Ensure model is in evaluation mode
total_test_loss_mse = 0
correct_test_predictions = 0
total_test_samples = 0
with torch.no_grad():
    for features, labels in test_loader: # Use test_loader here
        features, labels = features.to(device), labels.to(device)
        predictions = model(features) # Model outputs continuous values (0-5)
        test_loss = criterion(predictions, labels)
        total_test_loss_mse += test_loss.item()

        # Accuracy calculation for test set
        rounded_predictions = torch.round(predictions)
        correct_test_predictions += (rounded_predictions == labels).sum().item()
        total_test_samples += labels.size(0)

avg_test_loss_mse = total_test_loss_mse / len(test_loader)
avg_test_loss_rmse = np.sqrt(avg_test_loss_mse)
test_accuracy = (correct_test_predictions / total_test_samples) * 100 if total_test_samples > 0 else 0

print(f'\nFINAL TEST SET PERFORMANCE:')
print(f'Test Loss (MSE): {avg_test_loss_mse:.4f}')
print(f'Test RMSE: {avg_test_loss_rmse:.4f}')
print(f'Test Accuracy (Rounded): {test_accuracy:.2f}%')

mlflow.log_metrics({
    "test_loss": avg_test_loss_mse,
    "test_accuracy": test_accuracy
})
