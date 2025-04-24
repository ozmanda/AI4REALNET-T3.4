import yaml
from typing import Dict

def load_config_file(filepath:str) -> Dict[str, str]:
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config


def edit_config_file(filepath: str, key: str, new_value: str, second_key: str = None) -> None:
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
        file.close()
    
    if second_key:
        config[key][second_key] = new_value
    else:
        config[key] = new_value

    with open(filepath, 'w') as file:
        yaml.dump(config, file)
        file.close()