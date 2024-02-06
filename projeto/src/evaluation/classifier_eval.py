import os
import sys
from typing import Any, List

import numpy as np
import pandas as pd
import structlog
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utils.utils import load_config_file

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
logger = structlog.getLogger()


class ModelEvaluation:
    """
    Classe para avaliação de modelos de machine learning.

    Esta classe fornece métodos para avaliar a performance de modelos de machine learning,
    incluindo validação cruzada e métricas de avaliação.

    Attributes:
        model (np.ndarray): O modelo de machine learning a ser avaliado.
        x (pd.DataFrame): O DataFrame contendo os recursos de entrada.
        y (pd.DataFrame): O DataFrame contendo os rótulos alvo.
        n_splits (int, optional): O número de divisões para a validação cruzada. Padrão é 5.
    """

    def __init__(
        self, model: Any, x: pd.DataFrame, y: pd.DataFrame, n_splits: int = 5
    ) -> None:
        """
        Inicializa uma instância da classe ModelEvaluation.

        Args:
            model: O modelo de machine learning a ser avaliado.
            x (pd.DataFrame): O DataFrame contendo os recursos de entrada.
            y (pd.DataFrame): O DataFrame contendo os rótulos alvo.
            n_splits (int, opcional): O número de divisões para a validação cruzada. Padrão é 5.
        """

        self.model = model
        self.x = x
        self.y = y
        self.n_splits = n_splits

    def cross_val_evaluate(self) -> List[float]:
        """
        Realiza a validação cruzada do modelo.

        Retorna uma lista contendo os valores de AUC-ROC para cada fold da validação cruzada.

        Returns:
            List[float]: Uma lista contendo os valores de AUC-ROC para cada fold da validação cruzada.
        """

        logger.info("Iniciou a validação cruzada.")
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=load_config_file().get("random_state"),
        )

        scores = cross_val_score(
            self.model, self.x, self.y, cv=skf, scoring="roc_auc"
        )
        return scores

    def roc_auc_scorer(self, model, x: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Calcula a métrica AUC-ROC para o modelo dado os dados de entrada e saída.

        Args:
            model: O modelo de machine learning.
            x (pd.DataFrame): O DataFrame contendo os recursos de entrada.
            y (pd.DataFrame): O DataFrame contendo os rótulos alvo.

        Returns:
            float: O valor da métrica AUC-ROC.
        """

        y_pred = model.predict_proba(x)[:, 1]
        return roc_auc_score(y, y_pred)

    @staticmethod
    def evaluate_predictions(y_true, y_pred_proba) -> float:
        """
        Avalia as predições do modelo usando a métrica AUC-ROC.

        Args:
            y_true: O array contendo os rótulos alvo verdadeiros.
            y_pred_proba: O array contendo as probabilidades preditas para as classes positivas.

        Returns:
            float: O valor da métrica AUC-ROC.
        """

        logger.info("Iniciou a validação do modelo.")
        return roc_auc_score(y_true, y_pred_proba)
