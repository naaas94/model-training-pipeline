#!/bin/bash

# Entrypoint script for ML Pipeline container

set -e

echo "Starting ML Pipeline..."

# Check if config file exists
if [ ! -f "$PIPELINE_CONFIG" ]; then
    echo "Error: Pipeline config file not found at $PIPELINE_CONFIG"
    exit 1
fi

# Check if GCS credentials exist
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "Warning: GCS credentials not found at $GOOGLE_APPLICATION_CREDENTIALS"
    echo "GCS integration will be disabled"
fi

# Check if data directory exists
if [ ! -d "/app/data" ]; then
    echo "Error: Data directory not found at /app/data"
    exit 1
fi

# Create output directories if they don't exist
mkdir -p /app/models /app/output /app/logs

echo "Environment check complete"
echo "Config file: $PIPELINE_CONFIG"
echo "GCS credentials: $GOOGLE_APPLICATION_CREDENTIALS"
echo "Data directory: /app/data"
echo "Output directory: /app/output"
echo "Models directory: /app/models"
echo "Logs directory: /app/logs"

# Execute the pipeline
exec python scripts/train_pipeline.py
