# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy your transformer model checkpoint into the container
# (ensure transformer_recommender.pth is in your project root)
COPY transformer_recommender.pth /app/

# (Optional) Set default environment variables
ENV MODEL_PATH=/app/transformer_recommender.pth

# Expose the FastAPI port
EXPOSE 8000

# Launch the FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]