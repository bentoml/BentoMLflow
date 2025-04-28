import bentoml
import numpy as np
import numpy.typing as npt
from bentoml.models import BentoModel
from pydantic import Field
from bentoml.validators import Shape, DType
from typing import Annotated

demo_image = bentoml.images.Image(python_version="3.11") \
    .python_packages("mlflow", "scikit-learn")

target_names = ['setosa', 'versicolor', 'virginica']

@bentoml.service(
    image=demo_image,
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class IrisClassifier:
    bento_model = BentoModel("iris:latest")

    def __init__(self):
        self.model = self.bento_model.load_model()

    @bentoml.api(batchable=True)
    def predict(
        self,
        input_data: Annotated[npt.NDArray[np.float64], Shape((-1, 4)), DType("float64")] = Field(default=[[0.1, 0.4, 0.2, 1.0]])
    ) -> np.ndarray:
        print(f"batch_size: {len(input_data)}")
        preds = self.model.predict(input_data)
        return [target_names[i] for i in preds]
