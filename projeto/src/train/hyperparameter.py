import os
import sys
from typing import Any, Dict, Tuple, Union

import joblib
import mlflow
import pandas as pd
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.imputation import MeanMedianImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from hyperopt import STATUS_OK, fmin, hp, tpe
from mlflow.models import MetricThreshold, infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from data.data_load import DataLoad
from data.data_preprocess import DataPreprocess
from data.data_transformation import DataTransformation
from data.data_validation import DataValidation
from evaluation.classifier_eval import ModelEvaluation
from train import TrainModels
from utils.utils import load_config_file


def load_data() -> pd.DataFrame:
    """
    Carrega o conjunto de dados de treinamento.

    Returns:
        dataFrame: O conjunto de dados carregado.
    """

    dl = DataLoad()
    dataFrame = dl.load_data("train_dataset_name")
    return dataFrame


def split_data(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide o conjunto de dados em conjuntos de treinamento e validação.

    Args:
        dataframe (pd.DataFrame): O conjunto de dados de entrada.

    Returns:
        Os conjuntos de dados divididos e as variáveis alvo.
    """

    dv = DataValidation()
    is_valid = dv.run(df)

    dt = DataTransformation(df)
    X_train, X_valid, y_train, y_valid = dt.train_test_spliting()
    return X_train, X_valid, y_train, y_valid


def define_pipeline() -> Pipeline:
    """
    Define o pipeline de pré-processamento.

    Returns:
        Pipeline: O pipeline de pré-processamento.
    """

    pipe = Pipeline(
        [
            (
                "imputer",
                MeanMedianImputer(
                    variables=load_config_file().get("vars_imputer")
                ),
            ),
            (
                "discretizer",
                EqualFrequencyDiscretiser(
                    variables=load_config_file().get("vars_discretizer")
                ),
            ),
            ("scaler", SklearnTransformerWrapper(StandardScaler())),
        ]
    )
    return pipe


def objective(params: Dict[str, Any]) -> Dict[str, Union[float, int]]:
    """
    Define a função objetivo para otimização de hiperparâmetros.

    Args:
        params (Dict[str, Any]): Os hiperparâmetros a serem otimizados.

    Returns:
        Dict[str, Union[float, int]]: O resultado da avaliação.
    """

    with mlflow.start_run(run_name="with_discretizer_hyperopt"):
        mlflow.set_tag("model_name", "lr_hyperopt")
        mlflow.log_params(params)

        preprocessdor = DataPreprocess(pipe)
        preprocessdor.train(X_train)

        X_train_processed = preprocessdor.transform(X_train)
        X_valid_processed = preprocessdor.transform(X_valid)

        path_preprocess = load_config_file().get("path_preprocess")
        joblib.dump(preprocessdor, path_preprocess)

        mlflow.log_artifact(path_preprocess)

        mlflow.log_params(
            params={
                "imputer": pipe["imputer"],
                "discretizer": pipe["discretizer"],
                "scaler": pipe["scaler"],
            }
        )

        model = LogisticRegression(**params)
        model_eval = ModelEvaluation(
            model, X_train_processed, y_train, n_splits=5
        )
        roc_auc_scores = model_eval.cross_val_evaluate()

        mlflow.log_metric("train_roc_auc", roc_auc_scores.mean())

        model.fit(X_train_processed, y_train)

        y_val_preds = model_eval.model.predict_proba(X_valid_processed)[:, 1]
        val_roc_auc = model_eval.evaluate_predictions(y_valid, y_val_preds)

        mlflow.log_metric("valid_roc_auc", val_roc_auc)

        candidate_model_uri = mlflow.sklearn.log_model(
            model, "lr_model"
        ).model_uri

        signature = infer_signature(X_valid_processed, y_valid)

        eval_data = X_valid_processed
        eval_data["label"] = y_valid

        thereshold = {
            "accuracy_score": MetricThreshold(
                threshold=0.1,
                min_absolute_change=0.05,
                min_relative_change=0.05,
                greater_is_better=True,
            )
        }

        baseline_model = DummyClassifier(strategy="uniform").fit(
            X_train_processed, y_train
        )
        baseline_model_uri = mlflow.sklearn.log_model(
            baseline_model, "baseline_model", signature=signature
        ).model_uri

        mlflow.evaluate(
            candidate_model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
            validation_thresholds=thereshold,
            baseline_model=baseline_model_uri,
        )

        mlflow.end_run()

        return {"loss": -roc_auc_scores.mean(), "status": STATUS_OK}


def optimize_hyperparameters() -> Dict[str, Any]:
    """
    Otimiza hiperparâmetros usando o algoritmo TPE.

    Returns:
        Dict[str, Any]: Os melhores hiperparâmetros encontrados.
    """

    search_space = {
        "warm_start": hp.choice("warm_start", [True, False]),
        "fit_intercept": hp.choice("fit_intercept", [True, False]),
        "tol": hp.uniform("tol", 0.00001, 0.0001),
        "C": hp.uniform("C", 0.05, 3),
        "solver": hp.choice("solver", ["newton-cg", "lbfgs", "liblinear"]),
        "max_iter": hp.choice("max_iter", range(100, 1000)),
        "multi_class": "auto",
        "class_weight": hp.choice("class_weight", [None, "balanced"]),
    }

    best_result = fmin(
        fn=objective, space=search_space, algo=tpe.suggest, max_evals=5
    )
    return best_result


def train(X_train: pd.DataFrame, X_valid: pd.DataFrame):
    """
    Treina os modelos.

    Args:
        X_train: Dados de treinamento.
        X_valid: Dados de validação.
    """

    tm = TrainModels(X_train, X_valid)
    tm.run()


if __name__ == "__main__":
    df = load_data()
    X_train, X_valid, y_train, y_valid = split_data(df)
    pipe = define_pipeline()
    best_hyperparameters = optimize_hyperparameters()

    train(X_train, y_train)
