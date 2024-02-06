import json
import sqlite3
from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests
import structlog

conn = sqlite3.connect("../../preds.db")
cursor = conn.cursor()
logger = structlog.getLogger()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("prob_loan")


class Predict:
    """
    Classe para realizar predições usando um modelo em um endpoint.

    Esta classe envia os dados para um endpoint do mlflow, obtém as predições e salva os resultados
    na base de dados.

    Attributes:
        dataframe (pd.DataFrame): O DataFrame contendo os dados a serem preditos.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Inicializa uma instância da classe Predict.

        Args:
            dataframe (pd.DataFrame): O DataFrame contendo os dados a serem preditos.
        """

        self.dataframe = dataframe
        self.endpoint = "http://127.0.0.1:5001/invocations"

    def run(self) -> pd.DataFrame:
        """
        Executa o processo de predição.

        Retorna um DataFrame contendo as probabilidades das predições.

        Returns:
            pd.DataFrame: Um DataFrame contendo as probabilidades das predições.
        """

        logger.info("inciando a predição.")
        to_inderence = {
            "dataframe_split": {
                "columns": self.dataframe.columns.to_list(),
                "data": self.dataframe.replace(np.nan, None).values.tolist(),
            }
        }
        response = requests.post(self.endpoint, json=to_inderence)
        logger.info("Predições finalizadas.")

        probabilities = np.array(
            json.loads(response.text).get("predictions", [])
        )[:, 1]

        df_probs = self._results(probabilities)
        self._capture_inputs_and_predictions(to_inderence, df_probs)

        logger.info("Resultados salvo na base de dados.")
        return df_probs

    def _capture_inputs_and_predictions(
        self, inputs: Dict[str, Dict[str, Any]], preds: pd.DataFrame
    ) -> None:
        """
        Captura os inputs e as predições e salva na base de dados.

        Args:
            inputs (Dict[str, Dict[str, Any]]): Dados de entrada para a predição.
            preds (pd.DataFrame): Probabilidades das predições.

        Returns:
            None
        """

        input_df = pd.DataFrame(
            inputs["dataframe_split"]["data"],
            columns=inputs["dataframe_split"]["columns"],
        )
        input_df["preds_prob"] = preds

        # armazena no database
        self._store_in_database(input_df)

    def _store_in_database(self, input_df: pd.DataFrame) -> None:
        """
        Armazena os dados na base de dados.

        Args:
            input_df (pd.DataFrame): DataFrame contendo os dados a serem armazenados.

        Returns:
            None
        """

        input_df.to_sql("predictions", conn, if_exists="append", index=False)
        conn.commit()
        conn.close()

    def _results(self, probabilities: np.array) -> pd.DataFrame:
        """
        Cria um DataFrame com as probabilidades das predições.

        Args:
            probabilities (np.array): Array contendo as probabilidades das predições.

        Returns:
            pd.DataFrame: Um DataFrame contendo as probabilidades das predições.
        """

        df_results = pd.DataFrame()
        logger.info("Salvando as probabilidades.")
        df_results["probabilities_default"] = probabilities
        return df_results
