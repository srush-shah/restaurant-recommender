#!/bin/bash

# Simple monitoring script for transform container
# Usage: ./monitor_transform.sh

CONTAINER_NAME="etl_transform"
INTERVAL=30  # seconds between checks

echo "Starting transform monitoring..."
echo "Checking status every $INTERVAL seconds"
echo "Press Ctrl+C to stop monitoring"

while true; do
  # Get timestamp
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
  
  # Check if container exists and is running
  if ! docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "[$TIMESTAMP] Container $CONTAINER_NAME not running."
    
    # Check if it exists but is stopped
    if docker ps -a -q -f name=$CONTAINER_NAME | grep -q .; then
      echo "[$TIMESTAMP] Container exists but is stopped. Exit code: $(docker inspect $CONTAINER_NAME --format='{{.State.ExitCode}}')"
      
      # Get last 10 lines of logs
      echo "Last 10 lines of logs:"
      docker logs --tail 10 $CONTAINER_NAME
    else
      echo "[$TIMESTAMP] Container does not exist."
    fi
    
    echo "Monitoring stopped."
    exit 1
  fi
  
  # Container is running - get stats
  echo "[$TIMESTAMP] Container stats:"
  docker stats $CONTAINER_NAME --no-stream --format "CPU: {{.CPUPerc}}, MEM: {{.MemUsage}} ({{.MemPerc}})"
  
  # Get last log line to see current progress
  echo "[$TIMESTAMP] Latest log:"
  docker logs --tail 5 $CONTAINER_NAME | tail -n 5
  
  echo "----------------------------------------"
  
  # Wait for next check
  sleep $INTERVAL
done 