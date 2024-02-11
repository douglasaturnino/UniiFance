import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

import sqlite3

import pandas as pd
from data.data_load import DataLoad
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from utils.utils import load_config_file


class ModelMonitoring:
    """
    Classe para monitoramento de modelo.

    Esta classe carrega os dados de previsão e os dados de treinamento, calcula métricas de
    monitoramento de modelo, e gera um relatório de monitoramento.

    Attributes:
        query (str): A consulta SQL para recuperar os dados de previsão.
    """

    def __init__(self):
        """
        Inicializa uma instância da classe ModelMonitoring.
        """

        self.query = "SELECT * FROM predictions"

    def get_pred_data(self) -> pd.DataFrame:
        """
        Obtém os dados de previsão do banco de dados.

        Returns:
            pd.DataFrame: O DataFrame contendo os dados de previsão.
        """

        conn = sqlite3.connect("preds.db")
        df_pred = pd.read_sql_query(self.query, conn)
        conn.close()
        return df_pred

    def get_training_data(self) -> pd.DataFrame:
        """
        Obtém os dados de treinamento.

        Returns:
            pd.DataFrame: O DataFrame contendo os dados de treinamento.
        """

        dl = DataLoad()
        df_train = dl.load_data("train_dataset_name")
        return df_train

    def run(self) -> None:
        """
        Executa o monitoramento do modelo, calcula métricas e gera um relatório.

        Returns:
            None
        """

        df_cur = self.get_pred_data()  # dados atuais
        df_ref = self.get_training_data().drop(
            load_config_file().get("target_name"), axis=1
        )  # dados referencia

        model_card = Report(
            metrics=[
                DatasetSummaryMetric(),
                DataDriftPreset(),
                DatasetMissingValuesMetric(),
            ]
        )

        model_card.run(reference_data=df_ref, current_data=df_cur)
        model_card.save_html("projeto/docs/model_monitoring_report.html")


mm = ModelMonitoring()
mm.run()
