#!/bin/bash

# Exit on any error
set -e

# Set environment variables
export RCLONE_CONTAINER=object-persist-project23

echo "Starting ETL pipeline..."

# Create required directories
mkdir -p data/raw data/als

# Run extract
echo "Step 1: Extracting data..."
docker compose -f docker-compose-als-etl.yaml run extract
if [ $? -ne 0 ]; then
    echo "Extract step failed"
    exit 1
fi

# Run transform
echo "Step 2: Transforming data..."
docker compose -f docker-compose-als-etl.yaml run transform
if [ $? -ne 0 ]; then
    echo "Transform step failed"
    exit 1
fi

# Run load
echo "Step 3: Loading data..."
docker compose -f docker-compose-als-etl.yaml run load
if [ $? -ne 0 ]; then
    echo "Load step failed"
    exit 1
fi

# Cleanup
echo "Cleaning up containers..."
docker compose -f docker-compose-als-etl.yaml down

echo "ETL pipeline completed successfully!" 