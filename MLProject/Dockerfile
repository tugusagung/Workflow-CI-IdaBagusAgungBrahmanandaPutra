# MLflow Project Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    joblib==1.3.2

# Expose MLflow port
EXPOSE 5000

# Command to run MLflow project
CMD ["mlflow", "run", ".", "--experiment-name", "docker-container-run"]