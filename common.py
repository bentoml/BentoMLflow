import bentoml
import numpy as np
import numpy.typing as npt
from pydantic import Field
from bentoml.validators import Shape, DType
from typing import Annotated


my_image = bentoml.images.PythonImage(python_version="3.11") \
    .python_packages("mlflow", "scikit-learn")

class MyInputParams(bentoml.IODescriptor):
    input_data: Annotated[npt.NDArray[np.float64], Shape((-1, 4)), DType("float64")] = Field(default=[[0.1, 0.4, 0.2, 1.0]])
    client_id: str
