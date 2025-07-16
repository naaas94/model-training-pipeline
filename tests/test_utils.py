import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training_pipeline.utils import load_config, get_logger

def test_load_config():
    config = load_config('config.yaml')
    assert 'data' in config
    assert 'model' in config
    assert 'preprocessing' in config
    assert 'output' in config
    assert 'logging' in config

def test_get_logger():
    logger = get_logger('test_logger')
    assert logger.name == 'test_logger'
    logger.info('Logger test message') 