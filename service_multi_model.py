import bentoml
import numpy as np
import numpy.typing as npt
from bentoml.models import BentoModel

from common import MyInputParams, my_image


@bentoml.service(
    image=my_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class IrisClassifier:
    bento_model_1 = BentoModel("iris:v1")
    bento_model_2 = BentoModel("iris:v2")

    def __init__(self):
        self.model_1 = bentoml.mlflow.load_model(self.bento_model_1)
        self.model_2 = bentoml.mlflow.load_model(self.bento_model_2)

    @bentoml.api(route="/v1/predict", input_spec=MyInputParams)
    def predict_1(
        self,
        input_data,
        client_id,
    ) -> np.ndarray:
        rv = self.model_1.predict(input_data)
        return np.asarray(rv)

    @bentoml.api(route="/v2/predict", input_spec=MyInputParams)
    def predict_2(
        self,
        input_data,
        client_id,
    ) -> np.ndarray:
        rv = self.model_2.predict(input_data)
        return np.asarray(rv)
    
    @bentoml.api(input_spec=MyInputParams)
    def predict_combined(
        self,
        input_data,
        client_id,
    ) -> np.ndarray:
        rv_a = self.model_1.predict(input_data)
        rv_b = self.model_2.predict(input_data)
        return np.asarray([rv_a, rv_b])
