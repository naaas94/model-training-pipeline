# utils.py: Utility functions for config loading and logging
import yaml
import logging

# Config loader
def load_config(config_path: str):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Logger setup
def get_logger(name: str):
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger 