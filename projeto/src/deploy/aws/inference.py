import json

import boto3
import pandas as pd


class Inference:
    """
    Classe para realizar inferência usando um modelo hospedado no Amazon SageMaker.

    Attributes:
        app_name (str): O nome do aplicativo SageMaker a ser consultado para inferência.
        region (str): A região AWS onde o aplicativo SageMaker está hospedado.
        df_test (pd.DataFrame): O DataFrame de teste a ser usado para inferência.

    Methods:
        query: Consulta o endpoint SageMaker para obter previsões com base nos dados de entrada.
    """

    def __init__(self) -> None:
        """
        Inicializa a classe de inferência.
        Carrega o DataFrame de teste a partir de um arquivo CSV local.
        """

        self.app_name = "prob-loan-sagemaker"
        self.region = "us-east-1"
        self.df_test = pd.read_csv("projeto/data/raw/test.csv")

    def query(self, input_json: json) -> json:
        """
        Consulta o endpoint SageMaker para obter previsões com base nos dados de entrada.

        Args:
            input_json (json): Dados de entrada em formato JSON a serem enviados para o endpoint SageMaker.

        Returns:
            json: As previsões retornadas pelo modelo hospedado no SageMaker.
        """

        client = boto3.session.Session().client(
            "sagemaker-runtime", self.region
        )
        response = client.invoke_endpoint(
            EndpointName=self.app_name,
            Body=input_json,
            ContentType="application/json",
        )
        preds = response["Body"].read().decode("ascii")
        preds = json.loads(preds)
        print(f"Resposta recebida: {preds}")
        return preds


if __name__ == "__main__":
    inference = Inference()
    # manipulacao
    data = {
        "dataframe_split": inference.df_test.iloc[[0]].to_dict(orient="split")
    }
    byte_data = json.dumps(data).encode("utf-8")

    output = inference.query(byte_data)

    resp = pd.DataFrame([output])
    print(resp)
