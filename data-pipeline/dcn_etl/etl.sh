#!/bin/bash
set -e

# Extract
docker compose -f docker-compose-dcn.yaml run extract

# Transform
docker compose -f docker-compose-dcn.yaml run transform

# Load
docker compose -f docker-compose-dcn.yaml run load