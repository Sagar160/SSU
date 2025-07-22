import os
import yaml

def read_yaml_config(filename):
    """
    Reads and returns the contents of a YAML config file as a dictionary.
    Raises FileNotFoundError if the file does not exist.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{filename}' does not exist in the current directory.")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)