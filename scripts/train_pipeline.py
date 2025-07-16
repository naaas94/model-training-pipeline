import sys
import os
from src.model_training_pipeline.utils import load_config, get_logger
from src.model_training_pipeline.data import load_data

CONFIG_PATH = os.environ.get('PIPELINE_CONFIG', 'config.yaml')

def main():
    config = load_config(CONFIG_PATH)
    logger = get_logger('train_pipeline')
    logger.info('Starting model training pipeline...')
    logger.info('Pipeline config loaded: %s', config)

    # Load and validate training data
    data_path = config['data']['path']
    logger.info(f'Loading training data from: {data_path}')
    df = load_data(data_path)
    logger.info(f'Final training data shape: {df.shape}')
    if df.empty:
        logger.error('No valid training data found. Exiting pipeline.')
        sys.exit(1)
    # Next pipeline steps will be added here

if __name__ == '__main__':
    main() 