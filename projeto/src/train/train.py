import os
import sys
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import structlog
import pandas as pd

from utils.utils import load_config_file
from utils.utils import save_model

logger = structlog.getLogger()

class TrainModels:
    """
    Classe para treinamento de modelos de machine learning.

    Esta classe fornece um método para treinar um modelo de machine learning com os dados fornecidos.

    Attributes:
        dados_X (pd.DataFrame): O DataFrame contendo os recursos de treinamento.
        dados_y (pd.DataFrame): O DataFrame contendo os rótulos alvo correspondentes aos dados de treinamento.
    """
    def __init__(self, dados_X: pd.DataFrame,
                       dados_y: pd.DataFrame):
        """
        Inicializa uma instância da classe TrainModels.

        Args:
            dados_X (pd.DataFrame): O DataFrame contendo os recursos de treinamento.
            dados_y (pd.DataFrame): O DataFrame contendo os rótulos alvo correspondentes aos dados de treinamento.
        """
        self.dados_X = dados_X 
        self.dados_y = dados_y

        
    def train(self, model) -> Any:
        """
        Treina o modelo de machine learning com os dados fornecidos.

        Args:
            model: O modelo de machine learning a ser treinado.

        Returns:
            model: O modelo treinado.
        """
        model.fit(self.dados_X, self.dados_y)
        save_model(model = model)
        return model 
    
