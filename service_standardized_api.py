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
    bento_model = BentoModel("iris:latest")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api(input_spec=MyInputParams)
    def predict(
        self,
        input_data,
        client_id,
    ) -> np.ndarray:
        print(f"processing request form user {client_id}")
        rv = self.model.predict(input_data)
        return np.asarray(rv)
