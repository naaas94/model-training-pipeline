"""
Model Registry for tracking model runs and metadata.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from .utils import get_logger

logger = get_logger('model_registry')

class ModelRegistry:
    """
    Registry for tracking model runs, metadata, and performance metrics.
    """
    
    def __init__(self, registry_path: str = 'models/model_registry.json'):
        self.registry_path = registry_path
        self.ensure_registry_exists()
    
    def ensure_registry_exists(self):
        """Ensure the registry file exists."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        if not os.path.exists(self.registry_path):
            self._save_registry({
                'runs': [],
                'models': {},
                'last_updated': datetime.now().isoformat()
            })
    
    def _load_registry(self) -> Dict:
        """Load the registry from file."""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {'runs': [], 'models': {}, 'last_updated': datetime.now().isoformat()}
    
    def _save_registry(self, registry: Dict):
        """Save the registry to file."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def log_run(self, run_data: Dict) -> str:
        """
        Log a new model training run.
        
        Args:
            run_data: Dictionary containing run information including:
                - model_version: str
                - embedding_model: str
                - metrics: dict (f1, accuracy, etc.)
                - hyperparameters: dict
                - data_info: dict (train_size, test_size, class_distribution)
                - gcs_paths: dict (model_path, metadata_path)
                - run_timestamp: str
                - status: str ('success', 'failed')
                - notes: str (optional)
        
        Returns:
            run_id: str
        """
        registry = self._load_registry()
        
        # Generate run ID
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add run to registry
        run_entry = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            **run_data
        }
        
        registry['runs'].append(run_entry)
        
        # Add model to models dict
        if run_data.get('model_version'):
            registry['models'][run_data['model_version']] = {
                'run_id': run_id,
                'timestamp': run_entry['timestamp'],
                'embedding_model': run_data.get('embedding_model'),
                'metrics': run_data.get('metrics', {}),
                'hyperparameters': run_data.get('hyperparameters', {}),
                'gcs_paths': run_data.get('gcs_paths', {}),
                'status': run_data.get('status', 'unknown')
            }
        
        registry['last_updated'] = datetime.now().isoformat()
        self._save_registry(registry)
        
        logger.info(f"Logged run {run_id} to registry")
        return run_id
    
    def get_latest_model(self) -> Optional[Dict]:
        """Get the latest successful model from the registry."""
        registry = self._load_registry()
        
        if not registry['runs']:
            return None
        
        # Find the latest successful run
        successful_runs = [run for run in registry['runs'] if run.get('status') == 'success']
        if not successful_runs:
            return None
        
        latest_run = max(successful_runs, key=lambda x: x['timestamp'])
        return latest_run
    
    def get_model_by_version(self, model_version: str) -> Optional[Dict]:
        """Get model information by version."""
        registry = self._load_registry()
        return registry['models'].get(model_version)
    
    def get_run_history(self, limit: int = 10) -> List[Dict]:
        """Get recent run history."""
        registry = self._load_registry()
        runs = registry['runs']
        return sorted(runs, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def export_registry_to_csv(self, output_path: str = 'models/model_registry.csv'):
        """Export registry to CSV for analysis."""
        registry = self._load_registry()
        
        if not registry['runs']:
            logger.warning("No runs in registry to export")
            return
        
        # Flatten run data for CSV
        flattened_runs = []
        for run in registry['runs']:
            flat_run = {
                'run_id': run['run_id'],
                'timestamp': run['timestamp'],
                'model_version': run.get('model_version'),
                'embedding_model': run.get('embedding_model'),
                'status': run.get('status'),
                'f1_score': run.get('metrics', {}).get('f1'),
                'accuracy': run.get('metrics', {}).get('accuracy'),
                'roc_auc': run.get('metrics', {}).get('roc_auc'),
                'train_size': run.get('data_info', {}).get('train_size'),
                'test_size': run.get('data_info', {}).get('test_size'),
                'notes': run.get('notes', '')
            }
            flattened_runs.append(flat_run)
        
        df = pd.DataFrame(flattened_runs)
        df.to_csv(output_path, index=False)
        logger.info(f"Registry exported to {output_path}")
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of model performance over time."""
        registry = self._load_registry()
        successful_runs = [run for run in registry['runs'] if run.get('status') == 'success']
        
        if not successful_runs:
            return {'message': 'No successful runs found'}
        
        f1_scores = [run.get('metrics', {}).get('f1') for run in successful_runs if run.get('metrics', {}).get('f1')]
        accuracy_scores = [run.get('metrics', {}).get('accuracy') for run in successful_runs if run.get('metrics', {}).get('accuracy')]
        
        return {
            'total_runs': len(registry['runs']),
            'successful_runs': len(successful_runs),
            'failed_runs': len(registry['runs']) - len(successful_runs),
            'avg_f1': sum(f1_scores) / len(f1_scores) if f1_scores else None,
            'avg_accuracy': sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else None,
            'best_f1': max(f1_scores) if f1_scores else None,
            'best_accuracy': max(accuracy_scores) if accuracy_scores else None,
            'latest_model_version': successful_runs[-1].get('model_version') if successful_runs else None
        } 