import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from utils.utils import load_config_file

logger = structlog.getLogger()

class DataTransformation:
    """
    Classe para realizar transformações nos dados, incluindo divisão em conjuntos de treino e teste.

    Esta classe oferece funcionalidades para manipulação de dados, incluindo a divisão do conjunto de dados em
    subconjuntos de treino e teste.

    Attributes:
        dataframe (pd.DataFrame): O DataFrame contendo os dados a serem transformados.
        target_name (str): O nome da coluna alvo no DataFrame.
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        Inicializa uma instância da classe DataTransformation.

        Args:
            dataframe (pd.DataFrame): O DataFrame contendo os dados a serem transformados.
        """
        self.dataframe = dataframe 
        self.target_name = load_config_file().get('target_name')
        
    def train_test_spliting(self):
        """
        Divide o conjunto de dados em subconjuntos de treino e teste.

        Retorna os conjuntos de treino e teste para recursos (X) e alvos (y).

        Returns:
            tuple: Uma tupla contendo quatro elementos: X_train, X_valid, y_train, y_valid.
                X_train (pd.DataFrame): O conjunto de recursos de treino.
                X_valid (pd.DataFrame): O conjunto de recursos de teste.
                y_train (pd.Series): O conjunto de alvos de treino.
                y_valid (pd.Series): O conjunto de alvos de teste.
        """
        X = self.dataframe.drop(self.target_name, axis=1)
        y = self.dataframe[self.target_name]
        
        X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                              y,
                                                              test_size=load_config_file().get('test_size'),
                                                              random_state=load_config_file().get('random_state'),
                                                              stratify=y)
        
        return X_train, X_valid, y_train, y_valid