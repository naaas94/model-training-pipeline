import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training_pipeline.utils import load_config

def test_load_config():
    config = load_config('config.yaml')
    assert 'data' in config
    assert 'model' in config 