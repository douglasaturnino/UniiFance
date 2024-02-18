import os
import sys
from typing import Dict

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

    Methods:
        check_shape_data: Verifica se as colunas do DataFrame correspondem à configuração definida.
        check_columns: Verifica se as colunas do DataFrame atendem aos critérios definidos.
        run: Executa as validações da estrutura e das colunas do DataFrame.
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
        config = load_config_file().get("columns")

        schema = self.create_dataframe_schema(config)

        try:
            schema.validate(dataframe)
            logger.info("Validation columns passed...")
            return True
        except pandera.errors.SchemaErrors as exc:
            logger.error("Validation columns failed...")
            pandera.display(exc.failure_cases)
        return False

    def create_dataframe_schema(self, config: Dict) -> DataFrameSchema:
        """
        Cria um esquema de DataFrame com base na configuração fornecida.

        Este método cria um DataFrameSchema com base na configuração passada. A configuração
        deve ser um dicionário contendo informações sobre as colunas do DataFrame.

        Args:
        - config (Dict): Um dicionário contendo informações sobre as colunas do DataFrame.
                         Espera-se que o dicionário tenha uma chave "columns" que contém uma lista
                         de dicionários, onde cada dicionário representa uma coluna do DataFrame.
                         Cada dicionário de coluna deve conter pelo menos as chaves "name" e "type",
                         que representam o nome e o tipo da coluna, respectivamente.

        Returns:
        - DataFrameSchema: Um esquema de DataFrame baseado na configuração fornecida.
                           O esquema define as expectativas de validação para cada coluna do DataFrame.

        Example:
        Para criar um esquema de DataFrame, suponha que a configuração seja a seguinte:

        config = {
            "columns": [
                {"name": "column1", "type": str, "nullable": True},
                {"name": "column2", "type": int, "nullable": False},
                {"name": "column3", "type": float, "nullable": True},
            ]
        }

        Este método irá iterar sobre as colunas especificadas na configuração e criar um esquema
        DataFrameSchema com base nessas informações. Por exemplo, ele criará validações para garantir
        que os tipos das colunas correspondam aos tipos especificados e que as colunas que são
        marcadas como não nulas não contenham valores nulos.

        """

        columns = {
            column_data["name"]: self.create_column(column_data)
            for column_data in config
        }
        return DataFrameSchema(columns)

    def create_column(self, column_data):
        """
        Cria uma instância de coluna com base nos dados fornecidos.

        Este método cria uma instância de coluna com base nos dados passados como argumento.
        Os dados devem conter informações sobre o tipo, coercibilidade, nulabilidade e verificações
        que devem ser aplicadas à coluna.

        Args:
        - column_data (Dict): Um dicionário contendo informações sobre a coluna a ser criada.
                              Espera-se que o dicionário contenha as seguintes chaves:
                              - "type": Tipo de dados da coluna. Por exemplo, str, int, float, etc.
                              - "coerce" (opcional): Um valor booleano indicando se a coerção deve ser aplicada.
                                                     Por padrão, é False se não estiver presente.
                              - "nullable" (opcional): Um valor booleano indicando se a coluna pode conter valores nulos.
                                                        Por padrão, é True se não estiver presente.
                              - "checks" (opcional): Uma lista de verificações que devem ser aplicadas à coluna.
                                                      Cada verificação é representada por um dicionário com uma das
                                                      seguintes chaves: "isin" ou "custom_check".
                                                      - "isin": Verifica se os valores da coluna estão presentes em
                                                                um conjunto específico de valores.
                                                                Espera-se que o valor associado seja uma lista
                                                                de valores permitidos.
                                                      - "custom_check": Permite definir uma verificação personalizada
                                                                        fornecendo uma expressão Python que será avaliada
                                                                        para cada valor na coluna.

        Returns:
        - Column: Uma instância de coluna criada com base nos dados fornecidos.

        Example:
        Para criar uma coluna, suponha que os dados fornecidos sejam os seguintes:

        column_data = {
            "type": str,
            "coerce": True,
            "nullable": False,
            "checks": [
                {"isin": ["A", "B", "C"]},
                {"custom_check": "lambda x: x.startswith('X')"}
            ]
        }

        Este método criará uma instância de coluna com o tipo de dados str, permitindo coerção,
        sem permitir valores nulos e aplicando duas verificações: uma para verificar se os valores
        estão em ["A", "B", "C"] e outra para verificar se os valores começam com 'X'.

        """
        checks = []
        for check_data in column_data.get("checks", []):
            if "isin" in check_data:
                checks.append(Check.isin((check_data["isin"])))
            elif "custom_check" in check_data:
                checks.append(Check(eval(check_data["custom_check"])))

        return Column(
            column_data["type"],
            coerce=column_data.get("coerce", False),
            nullable=column_data.get("nullable", True),
            checks=checks,
        )

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
