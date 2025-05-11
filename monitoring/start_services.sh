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

# Start MLflow
echo "Starting MLflow tracking services..."
docker compose -f ~/restaurant-recommender/ml_train_docker/docker-compose-mlflow.yaml up -d

echo "Waiting for MLflow to start..."
max_attempts=30
attempt=0
mlflow_ready=false

while [ $attempt -lt $max_attempts ]; do
  if docker ps | grep -q mlflow; then
    if curl -s http://localhost:5000/api/2.0/mlflow/experiments/list > /dev/null; then
      mlflow_ready=true
      echo "MLflow is ready!"
      break
    fi
  fi
  attempt=$((attempt+1))
  echo "Waiting for MLflow... (attempt $attempt/$max_attempts)"
  sleep 2
done

if [ "$mlflow_ready" = false ]; then
  echo "MLflow failed to start or respond in time, but continuing with other services..."
fi

# Build and start FastAPI and Prometheus
echo "Building FastAPI server..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml build fastapi_server

echo "Starting FastAPI and Prometheus..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml up -d fastapi_server prometheus

echo "Waiting for Prometheus to start..."
max_attempts=30
attempt=0
prometheus_ready=false

while [ $attempt -lt $max_attempts ]; do
  if docker ps | grep -q prometheus; then
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

# Start Grafana
echo "Starting Grafana..."
docker compose -f ~/restaurant-recommender/monitoring/docker-compose-fastapi.yaml up -d grafana

# Start Airflow
echo "Starting Airflow services..."
docker compose -f ~/restaurant-recommender/ml_train_docker/docker-compose-airflow.yaml up -d

echo "Waiting for Airflow Webserver to be ready..."
max_attempts=30
attempt=0
airflow_ready=false

while [ $attempt -lt $max_attempts ]; do
  if curl -s http://localhost:8081/health > /dev/null; then
    airflow_ready=true
    echo "Airflow Webserver is ready!"
    break
  fi
  attempt=$((attempt+1))
  echo "Waiting for Airflow Webserver... (attempt $attempt/$max_attempts)"
  sleep 2
done

if [ "$airflow_ready" = false ]; then
  echo "Airflow Webserver failed to start or respond in time"
  docker compose -f ~/restaurant-recommender/ml_train_docker/docker-compose-airflow.yaml logs airflow-webserver
fi

# Final Checks
echo "Verifying services are running..."
for service in mlflow minio postgres fastapi_server prometheus grafana airflow-webserver airflow-scheduler; do
  docker ps | grep $service || echo "$service not running!"
done

echo "Checking FastAPI health..."
curl -v http://localhost:8000/health || echo "Health endpoint not reachable"

echo "Docker network information:"
docker network ls
docker network inspect fastapi_test_monitoring_network || echo "Network not found"

# Wait for Grafana to be ready
if docker ps | grep -q grafana; then
  echo "Waiting for Grafana to be ready..."
  max_attempts=30
  attempt=0
  grafana_ready=false

  while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:3000/api/health > /dev/null; then
      grafana_ready=true
      echo "Grafana is ready!"
      break
    fi
    attempt=$((attempt+1))
    echo "Waiting for Grafana... (attempt $attempt/$max_attempts)"
    sleep 2
  done

  if [ "$grafana_ready" = false ]; then
    echo "Grafana failed to start or respond in time"
  fi
fi

echo "✅ All services processed!"
echo ""
echo "Access your services:"
echo "- MLflow:     http://localhost:5000"
echo "- FastAPI:    http://localhost:8000"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana:    http://localhost:3000 (admin/admin)"
echo "- Airflow:    http://localhost:8081 (admin/airflow)"
echo "- MinIO:      http://localhost:9001"
