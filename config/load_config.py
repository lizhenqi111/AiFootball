import yaml

def load_config(path):
    """加载YAML配置文件"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)