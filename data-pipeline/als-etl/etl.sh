#!/bin/bash

# Exit on any error
set -e

# Set environment variables
export RCLONE_CONTAINER=object-persist-project23

# Configure timeout (in seconds, default 2 hours)
TRANSFORM_TIMEOUT=7200

echo "Starting ETL pipeline..."

# Create required directories
mkdir -p data/raw data/als

# Clean up any orphaned containers first
echo "Cleaning up any orphaned containers..."
docker compose -f docker-compose-als-etl.yaml down --remove-orphans

# Run extract
echo "Step 1: Extracting data..."
docker compose -f docker-compose-als-etl.yaml run --rm extract
if [ $? -ne 0 ]; then
    echo "Extract step failed"
    exit 1
fi

# Run transform with timeout
echo "Step 2: Transforming data..."
echo "This may take a while. Timeout set to $TRANSFORM_TIMEOUT seconds ($(($TRANSFORM_TIMEOUT/60)) minutes)"

# Remove any orphaned containers again before transform step
docker compose -f docker-compose-als-etl.yaml down --remove-orphans

# Run transform with timeout
timeout $TRANSFORM_TIMEOUT docker compose -f docker-compose-als-etl.yaml run --rm transform

TRANSFORM_EXIT_CODE=$?
if [ $TRANSFORM_EXIT_CODE -eq 124 ]; then
    echo "Transform step timed out after $TRANSFORM_TIMEOUT seconds"
    # Cleanup the timed-out container
    echo "Cleaning up timed-out transform container..."
    docker compose -f docker-compose-als-etl.yaml down --remove-orphans
    exit 1
elif [ $TRANSFORM_EXIT_CODE -ne 0 ]; then
    echo "Transform step failed with exit code $TRANSFORM_EXIT_CODE"
    exit 1
fi

# Run load
echo "Step 3: Loading data..."
docker compose -f docker-compose-als-etl.yaml run --rm load
if [ $? -ne 0 ]; then
    echo "Load step failed"
    exit 1
fi

# Cleanup
echo "Cleaning up containers..."
docker compose -f docker-compose-als-etl.yaml down --remove-orphans

echo "ETL pipeline completed successfully!" 