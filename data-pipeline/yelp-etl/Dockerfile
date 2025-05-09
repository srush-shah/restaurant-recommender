FROM apache/airflow:2.7.3-python3.10

# Switch to root to install system packages
USER root

# Install system tools and rclone
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    git \
    rclone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user BEFORE installing Python packages
USER airflow

# Install Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Create working dirs
RUN mkdir -p /opt/airflow/scripts /opt/airflow/data

# Copy DAGs and scripts into container
COPY dags/ /opt/airflow/dags/
COPY scripts/ /opt/airflow/scripts/
