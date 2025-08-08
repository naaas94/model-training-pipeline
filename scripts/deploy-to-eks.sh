#!/bin/bash

# Deploy ML Pipeline to EKS

set -e

NAMESPACE="ml-pipeline"
KUBECONFIG="${KUBECONFIG:-~/.kube/config}"

echo "Deploying ML Pipeline to EKS..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if namespace exists, create if not
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    echo "Creating namespace: $NAMESPACE"
    kubectl apply -f k8s/namespace.yaml
fi

# Apply all Kubernetes manifests
echo "Applying Kubernetes manifests..."

# Apply ConfigMap and Secret
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# Apply ServiceAccount and RBAC
kubectl apply -f k8s/service-account.yaml

# Apply Persistent Volume Claims
kubectl apply -f k8s/persistent-volume-claim.yaml

# Apply Deployment
kubectl apply -f k8s/deployment.yaml

# Apply CronJob (optional - for scheduled runs)
kubectl apply -f k8s/cronjob.yaml

echo "Deployment complete!"
echo ""
echo "To check the status:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl logs -f deployment/ml-pipeline -n $NAMESPACE"
echo ""
echo "To trigger a manual run:"
echo "  kubectl create job --from=cronjob/ml-pipeline-cronjob manual-run -n $NAMESPACE"
echo ""
echo "To view logs:"
echo "  kubectl logs -f job/manual-run -n $NAMESPACE"
