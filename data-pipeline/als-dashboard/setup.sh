#!/bin/bash

# Create the Streamlit secrets directory if it doesn't exist
mkdir -p .streamlit

# Create the secrets.toml file with the provided credentials
cat > .streamlit/secrets.toml << EOL
# Chameleon Object Storage Credentials
application_credential_id = "aba2c756cb1649b6bd47c572f67a9528"
application_credential_secret = "HnUZC1Dvhnw1A5-19LiDQORiWURpL1IflqK3tk506NXJp8Tb0HSW4m0V9Ch4zEhwV9WR7tTxiAPTcTGHMi2SKw"
user_id = "59b49300deb6f11832f7764d9f2c3451ba51e578bb556d58e02bf4c64bd89a31"
EOL

echo "Credentials have been set up in .streamlit/secrets.toml"
echo "Building and starting the dashboard..."

# Build and start the dashboard using Docker Compose
docker compose up --build -d

echo ""
echo "Dashboard is now running at http://localhost:8501"
echo "To stop the dashboard, run: docker-compose down" 