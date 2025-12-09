import yaml
from easydict import EasyDict as edict
import os

def load_config(config_path):
    """
    Loads a YAML configuration file, handling inheritance with '_base_'.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if '_base_' in config:
        base_path = os.path.join(os.path.dirname(config_path), config['_base_'])
        base_config = load_config(base_path)
        del config['_base_']
        # Merge base config first, then overlay with current config
        merged_config = {**base_config, **config}
    else:
        merged_config = config
    
    return edict(merged_config) # Use EasyDict for dot notation access