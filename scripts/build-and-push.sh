#!/bin/bash

# Build and push script for ML Pipeline Docker image

set -e

# Configuration
IMAGE_NAME="ml-pipeline"
REGISTRY="${REGISTRY:-your-registry}"  # Set your registry URL
TAG="${TAG:-latest}"
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE_NAME}"

# Build the image
docker build -t ${FULL_IMAGE_NAME} .

echo "Image built successfully"

# Push to registry (uncomment when ready)
# echo "Pushing to registry..."
# docker push ${FULL_IMAGE_NAME}

echo "Build complete!"
echo "To push to registry, run: docker push ${FULL_IMAGE_NAME}"
echo "To deploy to EKS, update the image name in k8s/deployment.yaml and k8s/cronjob.yaml"
