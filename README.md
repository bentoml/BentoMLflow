<div align="center">
    <h1 align="center">Serve an MLflow Model with BentoML</h1>
</div>

This is a BentoML example project, which demonstrates how to serve and deploy an [MLflow](https://github.com/mlflow/mlflow) model with BentoML.

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoMLflow.git
cd BentoMLflow

# Recommend Python 3.11
pip install -r requirements.txt
```

## Train and save a model

Save the model to the BentoML Model Store:

```bash
python3 save_model.py
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

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

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html).

```bash
bentoml cloud login
```

Deploy it from the project directory.

```bash
bentoml deploy .
```

Once the application is up and running, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).
