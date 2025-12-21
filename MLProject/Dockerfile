# Use Python 3.12.7 slim image
FROM python:3.12.7-slim

# Set working directory
WORKDIR /app

# Copy model files
COPY outputs/ /app/outputs/
COPY modelling.py /app/

# Install dependencies (compatible versions for Python 3.12.7)
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    pandas \
    numpy \
    scikit-learn \
    joblib

# Expose port for MLflow (if needed)
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the model
CMD ["python", "modelling.py", "--experiment_name", "docker-run", "--n_estimators", "100", "--max_depth", "10"]