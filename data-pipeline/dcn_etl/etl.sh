#!/bin/bash

# Exit on any error
set -e

# Set environment variable if not already set
if [ -z "$RCLONE_CONTAINER" ]; then
    export RCLONE_CONTAINER=object-persist-project23
fi

echo "Starting DCN ETL pipeline..."

# Create required directories
mkdir -p data/raw data/user_latent_vectors data/item_latent_vectors data/dcn

# Run extract
echo "Step 1: Extracting data..."
docker compose -f docker-compose-dcn-etl.yaml run extract
if [ $? -ne 0 ]; then
    echo "Extract step failed"
    exit 1
fi

# Run transform
echo "Step 2: Transforming data..."
docker compose -f docker-compose-dcn-etl.yaml run transform
if [ $? -ne 0 ]; then
    echo "Transform step failed"
    exit 1
fi

# Run load
echo "Step 3: Loading data..."
docker compose -f docker-compose-dcn-etl.yaml run load
if [ $? -ne 0 ]; then
    echo "Load step failed"
    exit 1
fi

# Cleanup
echo "Cleaning up containers..."
docker compose -f docker-compose-dcn-etl.yaml down

echo "DCN ETL pipeline completed successfully!"