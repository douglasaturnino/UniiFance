import os
import sys

import pandas as pd
import pandera
import structlog
from pandera import Check, Column, DataFrameSchema
from utils.utils import load_config_file

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
logger = structlog.getLogger()


class DataValidation:
    """
    Classe de validação dos dados.

    Esta classe oferece funcionalidades para validar a estrutura e as colunas de um DataFrame.

    Attributes:
        columns_to_use (list): Lista de colunas a serem utilizadas no DataFrame.
    """

    def __init__(self) -> None:
        """
        Inicializa uma instância da classe DataValidation.
        """
        self.columns_to_use = load_config_file().get("columns_to_use")

    def check_shape_data(self, dataframe: pd.DataFrame) -> bool:
        """
        Verifica se as colunas do DataFrame correspondem à configuração definida.

        Args:
            dataframe (pd.DataFrame): DataFrame a ser validado.

        Returns:
            bool: True se a validação for bem-sucedida, False caso contrário.
        """
        try:
            logger.info("Validacao iniciou")
            dataframe.columns = self.columns_to_use
            return True
        except Exception as e:
            logger.error(f"Validacao errou: {e}")
            return False

    def check_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Verifica se as colunas do DataFrame atendem aos critérios definidos.

        Args:
            dataframe (pd.DataFrame): DataFrame a ser validado.

        Returns:
            bool: True se a validação for bem-sucedida, False caso contrário.
        """
        schema = DataFrameSchema(
            {
                "target": Column(
                    int,
                    Check.isin([0, 1]),
                    Check(lambda x: x > 0),
                    coerce=True,
                ),
                "TaxaDeUtilizacaoDeLinhasNaoGarantidas": Column(
                    float, nullable=True
                ),
                "Idade": Column(int, nullable=True),
                "NumeroDeVezes30-59DiasAtrasoNaoPior": Column(
                    int, nullable=True
                ),
                "TaxaDeEndividamento": Column(float, nullable=True),
                "RendaMensal": Column(float, nullable=True),
                "NumeroDeLinhasDeCreditoEEmprestimosAbertos": Column(
                    int, nullable=True
                ),
                "NumeroDeVezes90DiasAtraso": Column(int, nullable=True),
                "NumeroDeEmprestimosOuLinhasImobiliarias": Column(
                    int, nullable=True
                ),
                "NumeroDeVezes60-89DiasAtrasoNaoPior": Column(
                    int, nullable=True
                ),
                "NumeroDeDependentes": Column(float, nullable=True),
            }
        )
        try:
            schema.validate(dataframe)
            logger.info("Validation columns passed...")
            return True
        except pandera.errors.SchemaErrors as exc:
            logger.error("Validation columns failed...")
            pandera.display(exc.failure_cases)
        return False

    def run(self, dataframe: pd.DataFrame) -> bool:
        """
        Executa as validações da estrutura e das colunas do DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame a ser validado.

        Returns:
            bool: True se todas as validações forem bem-sucedidas, False caso contrário.
        """
        if self.check_shape_data(dataframe) and self.check_columns(dataframe):
            logger.info("Validação com sucesso.")
            return True
        else:
            logger.error("Validação falhou.")
            return False
