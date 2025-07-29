#!/usr/bin/env python3
"""
CLI tool for managing model registry and viewing model information.
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Add the project root to Python path so we can import from src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model_training_pipeline.model_registry import ModelRegistry

def main():
    parser = argparse.ArgumentParser(description='Model Registry Management Tool')
    parser.add_argument('command', choices=['list', 'latest', 'summary', 'export', 'get'], 
                       help='Command to execute')
    parser.add_argument('--version', help='Model version (for get command)')
    parser.add_argument('--limit', type=int, default=10, help='Number of runs to show (for list)')
    parser.add_argument('--output', help='Output file for export')
    
    args = parser.parse_args()
    
    registry = ModelRegistry()
    
    if args.command == 'list':
        print("📋 Recent Model Runs:")
        print("=" * 80)
        runs = registry.get_run_history(args.limit)
        
        for run in runs:
            print(f"Run ID: {run['run_id']}")
            print(f"Timestamp: {run['timestamp']}")
            print(f"Model Version: {run.get('model_version', 'N/A')}")
            print(f"Status: {run.get('status', 'unknown')}")
            print(f"Embedding Model: {run.get('embedding_model', 'N/A')}")
            
            metrics = run.get('metrics', {})
            if metrics:
                print(f"F1 Score: {metrics.get('f1', 'N/A'):.4f}")
                print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            
            print(f"Notes: {run.get('notes', 'N/A')}")
            print("-" * 80)
    
    elif args.command == 'latest':
        print("🔍 Latest Successful Model:")
        print("=" * 80)
        latest = registry.get_latest_model()
        
        if latest:
            print(f"Run ID: {latest['run_id']}")
            print(f"Model Version: {latest.get('model_version', 'N/A')}")
            print(f"Timestamp: {latest['timestamp']}")
            print(f"Embedding Model: {latest.get('embedding_model', 'N/A')}")
            
            metrics = latest.get('metrics', {})
            if metrics:
                print(f"F1 Score: {metrics.get('f1', 'N/A'):.4f}")
                print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                print(f"ROC AUC: {metrics.get('roc_auc', 'N/A')}")
            
            gcs_paths = latest.get('gcs_paths', {})
            if gcs_paths:
                print(f"Model GCS Path: {gcs_paths.get('model_gcs_path', 'N/A')}")
                print(f"Metadata GCS Path: {gcs_paths.get('metadata_gcs_path', 'N/A')}")
            
            print(f"Notes: {latest.get('notes', 'N/A')}")
        else:
            print("No successful models found in registry.")
    
    elif args.command == 'summary':
        print("📊 Model Performance Summary:")
        print("=" * 80)
        summary = registry.get_performance_summary()
        
        if 'message' in summary:
            print(summary['message'])
        else:
            print(f"Total Runs: {summary['total_runs']}")
            print(f"Successful Runs: {summary['successful_runs']}")
            print(f"Failed Runs: {summary['failed_runs']}")
            print(f"Average F1 Score: {summary['avg_f1']:.4f}" if summary['avg_f1'] else "Average F1 Score: N/A")
            print(f"Average Accuracy: {summary['avg_accuracy']:.4f}" if summary['avg_accuracy'] else "Average Accuracy: N/A")
            print(f"Best F1 Score: {summary['best_f1']:.4f}" if summary['best_f1'] else "Best F1 Score: N/A")
            print(f"Best Accuracy: {summary['best_accuracy']:.4f}" if summary['best_accuracy'] else "Best Accuracy: N/A")
            print(f"Latest Model Version: {summary['latest_model_version']}")
    
    elif args.command == 'export':
        output_file = args.output or 'models/model_registry_export.csv'
        registry.export_registry_to_csv(output_file)
        print(f"✅ Registry exported to {output_file}")
    
    elif args.command == 'get':
        if not args.version:
            print("❌ Error: --version is required for 'get' command")
            sys.exit(1)
        
        print(f"🔍 Model Details for Version: {args.version}")
        print("=" * 80)
        model_info = registry.get_model_by_version(args.version)
        
        if model_info:
            print(f"Run ID: {model_info['run_id']}")
            print(f"Timestamp: {model_info['timestamp']}")
            print(f"Embedding Model: {model_info.get('embedding_model', 'N/A')}")
            print(f"Status: {model_info.get('status', 'unknown')}")
            
            metrics = model_info.get('metrics', {})
            if metrics:
                print(f"F1 Score: {metrics.get('f1', 'N/A'):.4f}")
                print(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                print(f"ROC AUC: {metrics.get('roc_auc', 'N/A')}")
            
            hyperparams = model_info.get('hyperparameters', {})
            if hyperparams:
                print("Hyperparameters:")
                for key, value in hyperparams.items():
                    print(f"  {key}: {value}")
            
            gcs_paths = model_info.get('gcs_paths', {})
            if gcs_paths:
                print(f"Model GCS Path: {gcs_paths.get('model_gcs_path', 'N/A')}")
                print(f"Metadata GCS Path: {gcs_paths.get('metadata_gcs_path', 'N/A')}")
        else:
            print(f"❌ Model version '{args.version}' not found in registry.")

if __name__ == '__main__':
    main() 