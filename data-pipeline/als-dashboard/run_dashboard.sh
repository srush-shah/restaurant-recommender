#!/bin/bash

set -e  # Exit on any error

# Ensure logs directory exists
mkdir -p logs

# Setup rclone configuration
echo "Setting up rclone configuration..."
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf << EOL
[chi_tacc]
type = swift
user_id = 59b49300deb6f11832f7764d9f2c3451ba51e578bb556d58e02bf4c64bd89a31
application_credential_id = aba2c756cb1649b6bd47c572f67a9528
application_credential_secret = HnUZC1Dvhnw1A5-19LiDQORiWURpL1IflqK3tk506NXJp8Tb0HSW4m0V9Ch4zEhwV9WR7tTxiAPTcTGHMi2SKw
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
EOL

# Test rclone configuration 
echo "Testing rclone configuration..."
if ! rclone listremotes | grep -q "chi_tacc:"; then
    echo "ERROR: chi_tacc remote not found in rclone configuration."
    exit 1
fi

echo "Checking access to chi_tacc object storage..."
CONTAINER="object-persist-project23"
if ! rclone lsd chi_tacc:$CONTAINER > /dev/null 2>&1; then
    echo "ERROR: Cannot access chi_tacc:$CONTAINER. Please check your credentials."
    exit 1
fi

echo "Checking access to als directory..."
if ! rclone ls chi_tacc:$CONTAINER/als > /dev/null 2>&1; then
    echo "WARNING: Cannot list files in chi_tacc:$CONTAINER/als"
    echo "This may be normal if the directory doesn't exist yet."
fi

# Stop any running containers
echo "Stopping any existing dashboard containers..."
docker compose down

# Build and start the dashboard
echo "Building and starting the dashboard..."
docker compose up --build -d

# Check if container started successfully
if ! docker compose ps | grep -q "dashboard"; then
    echo "ERROR: Container failed to start. Checking logs..."
    docker compose logs
    exit 1
fi

echo "Dashboard is running at http://localhost:8501"
echo "To view logs: docker compose logs -f"