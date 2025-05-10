#!/bin/bash
set -e

export RCLONE_CONTAINER=object-persist-project23

# Extract
docker compose -f docker-compose-dcn.yaml run extract

# Transform
docker compose -f docker-compose-dcn.yaml run transform

# Load
docker compose -f docker-compose-dcn.yaml run load