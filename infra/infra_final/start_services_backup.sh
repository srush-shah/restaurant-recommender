#!/bin/bash

# Exit on error
set -e

echo "Starting services..."

# Ensure provisioning directories exist
echo "Checking Grafana provisioning directories..."
mkdir -p ~/restaurant-recommender/monitoring/grafana-provisioning/dashboards
mkdir -p ~/restaurant-recommender/monitoring/grafana-provisioning/datasources

# Ensure provisioning files exist
echo "Checking Grafana provisioning files..."
DASHBOARD_FILE=~/restaurant-recommender/monitoring/grafana-provisioning/dashboards/dashboard.yaml
DATASOURCE_FILE=~/restaurant-recommender/monitoring/grafana-provisioning/datasources/prometheus.yaml

# Create dashboard.yaml if it doesn't exist
if [ ! -f "$DASHBOARD_FILE" ]; then
  echo "Creating dashboard provisioning file..."
  cat > "$DASHBOARD_FILE" << 'EOF'
apiVersion: 1
providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 5
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: false
EOF
fi

# Create datasource.yaml if it doesn't exist
if [ ! -f "$DATASOURCE_FILE" ]; then
  echo "Creating datasource provisioning file..."
  cat > "$DATASOURCE_FILE" << 'EOF'
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "5s"
      queryTimeout: "60s"
      httpMethod: "POST"
EOF
fi

# Clean up existing containers
echo "Cleaning up existing containers..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml down --remove-orphans || true
docker compose -f ~/restaurant-recommender/ml_train_docker/docker-compose-mlflow.yaml down --remove-orphans || true
docker compose -f ~/restaurant-recommender/ml_train_docker/docker-compose-airflow.yaml down --remove-orphans || true
docker compose -f ~/restaurant-recommender/ml_train_docker/docker-compose-dashboard.yaml down --remove-orphans || true

# Start MLflow services
echo "Starting MLflow tracking services..."
docker compose -f ~/restaurant-recommender/ml_train_docker/docker-compose-mlflow.yaml up -d

# Wait for MLflow
echo "Waiting for MLflow to start..."
for i in {1..30}; do
  if docker ps | grep -q mlflow && curl -s http://localhost:5000/api/2.0/mlflow/experiments/list > /dev/null; then
    echo "MLflow is ready!"
    break
  fi
  echo "Waiting for MLflow... ($i/30)"
  sleep 2
done

# Start MinIO explicitly if needed
echo "Starting MinIO..."
docker compose -f ~/restaurant-recommender/ml_train_docker/docker-compose-mlflow.yaml up -d minio

# Start Airflow
echo "Starting Airflow..."
docker compose -f ~/restaurant-recommender/ml_train_docker/docker-compose-airflow.yaml up -d

# Build FastAPI
echo "Building FastAPI server..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml build fastapi_server

# Start FastAPI + Prometheus
echo "Starting FastAPI and Prometheus..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml up -d fastapi_server prometheus

# Wait for Prometheus
echo "Waiting for Prometheus..."
for i in {1..30}; do
  if docker ps | grep -q prometheus && curl -s http://localhost:9090/-/ready > /dev/null; then
    echo "Prometheus is ready!"
    break
  fi
  echo "Waiting for Prometheus... ($i/30)"
  sleep 2
done

# Start Grafana
echo "Starting Grafana..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml up -d grafana

# Start Streamlit dashboard
echo "Starting Streamlit dashboard..."
docker compose -f ~/restaurant-recommender/ml_train_docker/docker-compose-dashboard.yaml up -d

# Verify running services
echo "Verifying services..."
for svc in mlflow minio postgres fastapi_server prometheus grafana airflow streamlit; do
  echo "$svc:"
  docker ps | grep "$svc" || echo "$svc not running!"
done

# Logs if something failed
for svc in mlflow fastapi_server prometheus grafana airflow streamlit; do
  if ! docker ps | grep -q "$svc"; then
    echo "$svc logs:"
    docker compose -f ~/restaurant-recommender/*/docker-compose-*.yaml logs "$svc"
  fi
done

# FastAPI health
if docker ps | grep -q fastapi_server; then
  echo "FastAPI health check:"
  curl -v http://localhost:8000/health || echo "Health endpoint not reachable"
fi

# Docker network info
echo "Docker network information:"
docker network ls
docker network inspect fastapi_test_monitoring_network || echo "Network not found"

# Grafana readiness
echo "Waiting for Grafana..."
for i in {1..30}; do
  if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "Grafana is ready!"
    break
  fi
  echo "Waiting for Grafana... ($i/30)"
  sleep 2
done

# Final summary
echo "✅ All services processed!"
echo "Access them at:"
echo "- MLflow:     http://localhost:5000"
echo "- MinIO:      http://localhost:9001"
echo "- FastAPI:    http://localhost:8000"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana:    http://localhost:3000 (admin/admin)"
echo "- Airflow:    http://localhost:8080"
echo "- Streamlit:  http://localhost:8501"
