import sys
import os
import time
from src.model_training_pipeline.utils import load_config, get_logger
from src.model_training_pipeline.data import load_data
from src.model_training_pipeline.preprocessing import balance_data, stratified_train_test_split
from src.model_training_pipeline.training import train_model
from src.model_training_pipeline.evaluation import evaluate_model
from src.model_training_pipeline.persistence import save_model_to_gcs
from src.model_training_pipeline.model_registry import ModelRegistry
import pandas as pd

CONFIG_PATH = os.environ.get('PIPELINE_CONFIG', 'config.yaml')

def main():
    start_time = time.time()
    config = load_config(CONFIG_PATH)
    logger = get_logger('train_pipeline')
    logger.info('Starting model training pipeline...')
    logger.info('Pipeline config loaded: %s', config)

    # Initialize model registry
    registry = ModelRegistry()
    
    try:
        # Load and validate training data
        data_path = config['data']['path']
        logger.info(f'Loading training data from: {data_path}')
        df = load_data(data_path)
        logger.info(f'Final training data shape: {df.shape}')
        if df.empty:
            logger.error('No valid training data found. Exiting pipeline.')
            sys.exit(1)

        # Preprocessing: Balance data if enabled
        if config['preprocessing']['balance']:
            logger.info('Balancing dataset...')
            df = balance_data(
                df, 
                label_col='intent', 
                method=config['preprocessing']['balance_method'],
                random_state=config['data']['random_state']
            )
            logger.info(f'Balanced dataset shape: {df.shape}')

        # Split data into train/test sets
        logger.info('Splitting data into train/test sets...')
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            df,
            label_col='intent',
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )
        logger.info(f'Train set: {X_train.shape}, Test set: {X_test.shape}')

        # Train model
        logger.info('Training model...')
        model_config = config['model']
        model_config['embedding_model'] = config['preprocessing']['embedding_model']
        
        model, metadata = train_model(
            X_train, 
            y_train, 
            model_config,
            model_dir=config['output']['model_dir'],
            model_name=config['output']['model_name'],
            metadata_name=config['output']['metadata_name']
        )
        logger.info(f'Model training completed. Version: {metadata["model_version"]}')

        # Save model to GCS
        logger.info('Saving model to GCS...')
        gcs_info = save_model_to_gcs(
            model, 
            metadata, 
            bucket_name='pcc-datasets',
            model_prefix='pcc-models'
        )
        logger.info(f'Model saved to GCS: {gcs_info["model_gcs_path"]}')

        # Evaluate model
        logger.info('Evaluating model...')
        # Generate case IDs for test set
        case_ids = [f'case_{i}' for i in range(len(y_test))]
        
        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            case_ids=case_ids,
            model_version=metadata['model_version'],
            embedding_model=config['preprocessing']['embedding_model'],
            output_path=config['output']['predictions_path'],
            metrics_path=config['output']['metrics_path'],
            monitor_log_path=config['output']['monitor_log_path'],
            notes='Pipeline training run'
        )
        
        logger.info(f'Model evaluation completed. F1 Score: {metrics["f1"]:.4f}')
        
        # Log run to registry
        run_data = {
            'model_version': metadata['model_version'],
            'embedding_model': config['preprocessing']['embedding_model'],
            'metrics': metrics,
            'hyperparameters': metadata.get('hyperparameters', {}),
            'data_info': {
                'train_size': X_train.shape[0],
                'test_size': X_test.shape[0],
                'class_distribution': {
                    'train': dict(pd.Series(y_train).value_counts()),
                    'test': dict(pd.Series(y_test).value_counts())
                }
            },
            'gcs_paths': gcs_info,
            'run_timestamp': metadata.get('trained_on'),
            'status': 'success',
            'notes': f'Pipeline run - F1: {metrics["f1"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}',
            'processing_duration_seconds': time.time() - start_time
        }
        
        run_id = registry.log_run(run_data)
        logger.info(f'Run logged to registry with ID: {run_id}')
        
        # Export registry to CSV for analysis
        registry.export_registry_to_csv()
        
        # Print performance summary
        summary = registry.get_performance_summary()
        logger.info(f'Performance summary: {summary}')
        
        logger.info('Pipeline completed successfully!')
        
    except Exception as e:
        logger.error(f'Pipeline failed: {e}')
        
        # Log failed run to registry
        run_data = {
            'model_version': None,
            'embedding_model': config['preprocessing']['embedding_model'],
            'metrics': {},
            'hyperparameters': {},
            'data_info': {},
            'gcs_paths': {},
            'run_timestamp': time.time(),
            'status': 'failed',
            'notes': f'Pipeline failed: {str(e)}',
            'processing_duration_seconds': time.time() - start_time
        }
        
        registry.log_run(run_data)
        raise

if __name__ == '__main__':
    main() 