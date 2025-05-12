FROM python:3.9-slim

WORKDIR /app

# Install system dependencies first
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create data cache directory with proper permissions
RUN mkdir -p /app/data_cache && \
    chmod 777 /app/data_cache && \
    chown -R nobody:nogroup /app/data_cache

# Copy application files
COPY . .

# Create entrypoint script
RUN echo '#!/bin/bash\n\
    set -e\n\
    \n\
    echo "Starting Streamlit app..."\n\
    exec streamlit run app.py --server.address 0.0.0.0 --server.port 8501' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Expose the Streamlit port
EXPOSE 8501

# Switch to non-root user
USER nobody

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 