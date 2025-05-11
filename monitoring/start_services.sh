#!/bin/bash

# Exit on error
set -e

echo "Starting services..."

# Clean up existing containers to avoid conflicts
echo "Cleaning up existing containers..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml down --remove-orphans || true

# Build the FastAPI server first 
echo "Building FastAPI server..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml build fastapi_server

# Start FastAPI and Prometheus first, but not Grafana
echo "Starting FastAPI and Prometheus..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml up -d fastapi_server prometheus

# Wait for Prometheus to start up
echo "Waiting for Prometheus to start..."
max_attempts=30
attempt=0
prometheus_ready=false

while [ $attempt -lt $max_attempts ]; do
  if docker ps | grep -q prometheus; then
    echo "Prometheus is running, checking if it's responding..."
    if curl -s http://localhost:9090/-/ready > /dev/null; then
      prometheus_ready=true
      echo "Prometheus is ready!"
      break
    fi
  fi
  
  attempt=$((attempt+1))
  echo "Waiting for Prometheus... (attempt $attempt/$max_attempts)"
  sleep 2
done

if [ "$prometheus_ready" = false ]; then
  echo "Prometheus failed to start or respond in time"
  docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml logs prometheus
  exit 1
fi

# Now start Grafana
echo "Starting Grafana..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml up -d grafana

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
  docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml logs fastapi_server
fi

if ! docker ps | grep -q prometheus; then
  echo "Prometheus logs:"
  docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml logs prometheus
fi

if ! docker ps | grep -q grafana; then
  echo "Grafana logs:"
  docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml logs grafana
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
echo "docker logs fastapi_server"
echo "docker logs prometheus"
echo "docker logs grafana"