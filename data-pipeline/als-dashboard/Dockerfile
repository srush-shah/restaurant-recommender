FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install rclone for accessing object storage
RUN curl https://rclone.org/install.sh | bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Create the .streamlit directory
RUN mkdir -p .streamlit

# Create startup script to verify rclone config
RUN echo '#!/bin/bash\n\
    echo "Verifying rclone configuration..."\n\
    if [ -f /root/.config/rclone/rclone.conf ]; then\n\
    echo "Rclone config found:"\n\
    cat /root/.config/rclone/rclone.conf\n\
    echo "Testing rclone connection..."\n\
    rclone listremotes\n\
    else\n\
    echo "WARNING: No rclone configuration found!"\n\
    fi\n\
    \n\
    # Start Streamlit\n\
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
    ' > /app/start.sh \
    && chmod +x /app/start.sh

EXPOSE 8501

# Command to run the application
CMD ["/app/start.sh"] 