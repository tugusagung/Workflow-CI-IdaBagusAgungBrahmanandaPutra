# Dockerfile untuk MLflow Model
FROM python:3.12.7-slim

WORKDIR /app

# Copy project files
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    joblib==1.3.2

# Expose port untuk MLflow serving
EXPOSE 5000

# Command untuk run model (bisa diserving)
CMD ["python", "-c", "print('Diabetes ML Model Container Ready')"]