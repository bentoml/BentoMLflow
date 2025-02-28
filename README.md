<div align="center">
    <h1 align="center">Serve an MLflow Model with BentoML</h1>
</div>

[MLflow](https://github.com/mlflow/mlflow) is an open-source platform, purpose-built to assist machine learning practitioners and teams in handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.

This is a BentoML example project that demonstrates how to serve and deploy an MLflow model with BentoML.

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Install dependencies

Clone the repository.

```bash
git clone https://github.com/bentoml/BentoMLflow.git
cd BentoMLflow
```

Install BentoML and the required dependencies for this project.

```bash
# Recommend Python 3.11
pip install bentoml mlflow scikit-learn
```

While the example uses scikit-learn for demo purposes, both MLflow and BentoML support a wide variety of frameworks, such as PyTorch, TensorFlow and XGBoost.

## Train and save a model

Save the MLflow model to the BentoML Model Store:

```bash
python3 save_model.py
```

Verify that the model has been successfully saved:

```bash
bentoml models list
```

## Run BentoML Services

In this project, we provide multiple BentoML Services for the MLflow model for different use cases:

- `service.py`: A basic BentoML Service that serves the MLflow model
- `service_io_validate.py`: Enforces input data validation
- `service_batching.py`: Enables adaptive batching to efficiently handle concurrent requests 

For larger teams collaborating on multiple models and projects, you can use the following examples to standardize ML service development.

- `service_standardized_api.py`: Uses the shared components in `common.py` and enforces environment dependencies and API specifications across multiple projects
- `service_multi_model.py`: Serves multiple models in a single Service

For more information about them, see the blog post [Building ML Pipelines with MLflow and BentoML](https://www.bentoml.com/blog/building-ml-pipelines-with-mlflow-and-bentoml).

Let's try the Service with the basic setup. Run `bentoml serve` to start it.

```python
$ bentoml serve service.py:IrisClassifier

2024-06-19T10:25:31+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:IrisClassifier" listening on http://localhost:3000 (Press CTRL+C to quit)
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -X 'POST' \
    'http://localhost:3000/predict' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "input_data": [
        [5.9, 3, 5.1, 1.8]
    ]
}'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.predict(
        input_data=[
            [5.9, 3, 5.1, 1.8]
        ],
    )
    print(result)
```

</details>

For detailed explanations, see [the BentoML documentation](https://docs.bentoml.com/en/latest/examples/mlflow.html).

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html).

```bash
bentoml cloud login
```

Deploy it from the project directory.

```bash
bentoml deploy service.py:IrisClassifier
```

Once the application is up and running, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).
