import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
import structlog
from sklearn.pipeline import Pipeline
from utils.utils import load_config_file

logger = structlog.getLogger()


class DataPreprocess:
    """
    Classe para pré-processamento de dados utilizando pipelines.

    Esta classe fornece métodos para treinar e aplicar um pipeline de pré-processamento
    aos dados fornecidos.

    Attributes:
        pipe (Pipeline): O pipeline de pré-processamento a ser aplicado aos dados.
        trained_pipe (Pipeline): O pipeline treinado após a execução do método train().
    """

    def __init__(self, pipe: Pipeline):
        """
        Inicializa uma instância da classe DataPreprocess.

        Args:
            pipe (Pipeline): O pipeline de pré-processamento a ser aplicado aos dados.
        """
        self.pipe = pipe
        self.trained_pipe = None

    def train(self, dataframe: pd.DataFrame):
        """
        Treina o pipeline de pré-processamento com os dados fornecidos.

        Args:
            dataframe (pd.DataFrame): O DataFrame contendo os dados de treinamento.
        """
        logger.info("Pré-processamento iniciou.")
        self.trained_pipe = self.pipe.fit(dataframe)
        logger.info("pré-processamento terminou")

    def transform(self, dataframe: pd.DataFrame):
        """
        Aplica o pipeline treinado aos dados fornecidos.

        Args:
            dataframe (pd.DataFrame): O DataFrame contendo os dados a serem transformados.

        Returns:
            pd.DataFrame: O DataFrame resultante após a transformação.
        """
        if self.trained_pipe is None:
            raise ValueError("Pipeline não foi trinado.")

        logger.info("Transformação dos dados com preprocessador iniciou.")
        data_preprocessed = self.trained_pipe.transform(dataframe)
        logger.info("Transformação dos dados com preprocessador terminou.")
        return data_preprocessed
