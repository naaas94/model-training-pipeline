# ML Pipeline Deployment Guide

This guide covers deploying the Privacy Intent Classification ML Pipeline to EKS.

## System Overview

The ML pipeline is a **privacy intent classification model training system** with the following components:

### Pipeline Architecture:
1. **Data Ingestion** - Loads CSV/Parquet data with 584-dimensional embeddings
2. **Data Validation** - Validates schema and embedding dimensions
3. **Preprocessing** - SMOTE balancing + stratified train/test split
4. **Model Training** - Logistic Regression with hyperparameter optimization
5. **Evaluation** - F1, Accuracy, ROC-AUC metrics
6. **Persistence** - Saves to GCS with versioning
7. **Registry** - Tracks all runs and metadata

### Trigger Logic:
- **Manual trigger**: `python scripts/train_pipeline.py`
- **Scheduled trigger**: CronJob runs daily at 2 AM UTC
- **Data-driven**: Requires data file in mounted volume
- **Config-driven**: All behavior controlled via ConfigMap

## Prerequisites

### 1. EKS Cluster
```bash
# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### 2. Docker Registry
- ECR (AWS) or other container registry
- Update image names in `k8s/deployment.yaml` and `k8s/cronjob.yaml`

### 3. GCS Service Account
- Create service account with GCS access
- Download JSON key file
- Base64 encode for Kubernetes Secret

### 4. IAM Role for EKS
- Create IAM role with GCS permissions
- Update `k8s/service-account.yaml` with role ARN

## Quick Start

### 1. Build and Push Docker Image
```bash
# Set your registry
export REGISTRY="your-account.dkr.ecr.region.amazonaws.com"

# Build and push
chmod +x scripts/build-and-push.sh
./scripts/build-and-push.sh
```

### 2. Prepare GCS Credentials
```bash
# Base64 encode your GCS service account key
cat your-service-account.json | base64 -w 0

# Update k8s/secret.yaml with the encoded key
```

### 3. Deploy to EKS
```bash
chmod +x scripts/deploy-to-eks.sh
./scripts/deploy-to-eks.sh
```

## Configuration

### Environment Variables
- `PIPELINE_CONFIG`: Path to configuration file
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to GCS credentials
- `PYTHONPATH`: Python module path

### Configuration File (ConfigMap)
The pipeline configuration is stored in a ConfigMap:
- Data paths and schema
- Model hyperparameters
- Preprocessing settings
- Output paths

### Secrets
- GCS service account key (base64 encoded)
- Other sensitive credentials

## Storage

### Persistent Volumes
- **Data Volume**: Input training data (10Gi)
- **Models Volume**: Saved models and registry (20Gi)
- **Output Volume**: Predictions and metrics (10Gi)
- **Logs Volume**: Pipeline logs (5Gi)

### GCS Integration
- Models automatically saved to GCS
- Versioned with timestamp-based naming
- Metadata stored alongside models

## Monitoring and Logging

### Health Checks
- Liveness probe: Python process check
- Readiness probe: Python import check

### Logging
- Application logs in `/app/logs/`
- Kubernetes pod logs
- GCS integration logs

### Monitoring Commands
```bash
# Check pod status
kubectl get pods -n ml-pipeline

# View logs
kubectl logs -f deployment/ml-pipeline -n ml-pipeline

# Check CronJob status
kubectl get cronjobs -n ml-pipeline

# View job logs
kubectl logs -f job/manual-run -n ml-pipeline
```

## Triggering Pipeline Runs

### Manual Trigger
```bash
# Create a one-time job
kubectl create job --from=cronjob/ml-pipeline-cronjob manual-run -n ml-pipeline

# View logs
kubectl logs -f job/manual-run -n ml-pipeline
```

### Scheduled Trigger
- CronJob runs daily at 2 AM UTC
- Concurrency policy prevents overlapping runs
- Keeps last 3 successful and failed job histories

### Data-Driven Trigger
- Place new data in mounted data volume
- Update ConfigMap with new data path
- Restart deployment or trigger manual run

## Troubleshooting

### Common Issues

1. **Image Pull Errors**
   ```bash
   # Check image name in deployment
   kubectl describe pod -n ml-pipeline
   ```

2. **GCS Authentication Errors**
   ```bash
   # Verify secret is properly encoded
   kubectl get secret ml-pipeline-secrets -n ml-pipeline -o yaml
   ```

3. **Storage Issues**
   ```bash
   # Check PVC status
   kubectl get pvc -n ml-pipeline
   ```

4. **Resource Limits**
   ```bash
   # Check resource usage
   kubectl top pods -n ml-pipeline
   ```

### Debug Commands
```bash
# Exec into pod
kubectl exec -it deployment/ml-pipeline -n ml-pipeline -- /bin/bash

# Check environment variables
kubectl exec deployment/ml-pipeline -n ml-pipeline -- env

# Test GCS connectivity
kubectl exec deployment/ml-pipeline -n ml-pipeline -- python -c "from google.cloud import storage; print('GCS OK')"
```

## Scaling and Performance

### Resource Requirements
- **CPU**: 1-2 cores
- **Memory**: 2-4 Gi
- **Storage**: 45Gi total (data + models + output + logs)

### Performance Optimization
- Use larger instances for faster training
- Increase memory for large datasets
- Use SSD storage for better I/O

### Horizontal Scaling
- Multiple replicas for parallel training
- Separate namespaces for different environments
- Resource quotas for cost control

## Security Considerations

### Network Security
- Pod security policies
- Network policies
- Service mesh integration (optional)

### Data Security
- Encrypted volumes
- GCS encryption
- Secret management

### Access Control
- RBAC for pod access
- IAM roles for AWS services
- Service account permissions

## Backup and Recovery

### Data Backup
- GCS models and metadata
- Persistent volume snapshots
- Registry exports

### Disaster Recovery
- Multi-region GCS storage
- EKS cluster backups
- Configuration backups

## Cost Optimization

### Resource Management
- Right-size resource requests/limits
- Use spot instances for training
- Implement resource quotas

### Storage Optimization
- Use appropriate storage classes
- Implement data lifecycle policies
- Clean up old models and logs

## Integration with Existing Systems

### CI/CD Integration
- GitHub Actions for automated builds
- ArgoCD for GitOps deployment
- Tekton for pipeline automation

### Monitoring Integration
- Prometheus metrics
- Grafana dashboards
- AlertManager notifications

### Data Pipeline Integration
- Apache Airflow for orchestration
- Kubeflow for ML workflows
- Custom operators for pipeline triggers

## Support and Maintenance

### Regular Maintenance
- Update base images
- Rotate credentials
- Monitor resource usage
- Clean up old jobs and logs

### Updates and Upgrades
- Rolling updates for zero downtime
- Blue-green deployments for testing
- Canary deployments for gradual rollout

---

For additional support, refer to the main README.md or contact the development team.
noteId: "553aeaf0744711f0820c3965f99e25b7"
tags: []

---

