#!/bin/bash

# Exit on error
set -e

echo "Starting services..."

# Clean up existing containers to avoid conflicts
echo "Cleaning up existing containers..."
docker compose -f docker-compose-fastapi.yaml down --remove-orphans || true

# Fix path in docker-compose file if needed
echo "Checking docker-compose file path..."
if grep -q "/home/cc" docker-compose-fastapi.yaml; then
  echo "Fixing context path in docker-compose file..."
  sed -i.bak "s|context: /home/cc/restaurant-recommender/monitoring|context: .|g" docker-compose-fastapi.yaml
  echo "Path fixed in docker-compose file"
fi

# Build the FastAPI server first 
echo "Building FastAPI server..."
docker compose -f docker-compose-fastapi.yaml build fastapi_server

# Start the whole stack with detailed logging
echo "Starting all services..."
docker compose -f docker-compose-fastapi.yaml up -d

# Verify services are running
echo "Verifying services are running..."
echo "FastAPI server:"
docker ps | grep fastapi_server || echo "FastAPI server not running!"
echo "Prometheus:"
docker ps | grep prometheus || echo "Prometheus not running!"
echo "Grafana:"
docker ps | grep grafana || echo "Grafana not running!"

# Show logs for any services that failed to start
if ! docker ps | grep -q fastapi_server; then
  echo "FastAPI server logs:"
  docker compose -f docker-compose-fastapi.yaml logs fastapi_server
fi

if ! docker ps | grep -q prometheus; then
  echo "Prometheus logs:"
  docker compose -f docker-compose-fastapi.yaml logs prometheus
fi

if ! docker ps | grep -q grafana; then
  echo "Grafana logs:"
  docker compose -f docker-compose-fastapi.yaml logs grafana
fi

# Check if the FastAPI health endpoint is available
if docker ps | grep -q fastapi_server; then
  echo "Checking FastAPI health..."
  curl -v http://localhost:8000/health || echo "Health endpoint not reachable"
fi

# Show docker network info
echo "Docker network information:"
docker network ls
docker network inspect fastapi_test_monitoring_network || echo "Network not found"

echo "All services have been processed!"
echo "Services should be available at:"
echo "- FastAPI: http://localhost:8000"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo ""
echo "To view logs for each service, run:"
echo "docker compose -f docker-compose-fastapi.yaml logs fastapi_server"
echo "docker compose -f docker-compose-fastapi.yaml logs prometheus"
echo "docker compose -f docker-compose-fastapi.yaml logs grafana" 