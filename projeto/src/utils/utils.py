import os
import yaml
import joblib
from typing import Dict

def load_config_file() -> Dict:
    """
    Carrega um arquivo de configuração YAML a partir de um caminho relativo definido.

    Returns:
        dict: Um objeto Python contendo as configurações carregadas do arquivo YAML.

    """
    diretoria_atual = os.path.dirname(os.path.abspath(__file__))

    caminho_relativo = os.path.join('..', '..', 'config', 'config.yaml')
    
    config_file_path = os.path.abspath(os.path.join(diretoria_atual, caminho_relativo))
    
    config_file = yaml.safe_load(open(config_file_path, 'rb'))

    return config_file

def save_model(model) -> None:
    diretoria_atual = os.path.dirname(os.path.abspath(__file__))

    caminho_relativo = os.path.join('..', '..', 'models', load_config_file().get('model_name'))

    model_path = os.path.join(diretoria_atual, caminho_relativo)

    joblib.dump(model, model_path)