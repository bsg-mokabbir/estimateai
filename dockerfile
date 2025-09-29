# Use an updated NVIDIA CUDA runtime for GPU inference
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip curl unzip git \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to avoid system-wide pip error
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside venv
RUN pip install --upgrade pip

# Install AWS CLI for S3 sync inside venv
RUN pip install --no-cache-dir awscli boto3

# Copy Python dependencies
COPY requirements.txt .

# Install Python dependencies inside venv
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code 
COPY development/ development/

# AWS arguments and environment variables
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

ARG S3_BUCKET
ARG DETECTION_MODEL_FOLDER
ARG PDF_CLASSIFIER_FOLDER
ARG RAW_LEGEND_FOLDER

# Create data directory structure
RUN mkdir -p /app/data

# Download files from S3 using environment variables
RUN aws s3 sync s3://$S3_BUCKET/$DETECTION_MODEL_FOLDER /app/data/$DETECTION_MODEL_FOLDER && \
    aws s3 sync s3://$S3_BUCKET/$PDF_CLASSIFIER_FOLDER /app/data/$PDF_CLASSIFIER_FOLDER && \
    aws s3 sync s3://$S3_BUCKET/$RAW_LEGEND_FOLDER /app/data/$RAW_LEGEND_FOLDER

# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "development/streamlit_app/main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
