import time
import random
import requests

from concurrent.futures import ThreadPoolExecutor
from sklearn.datasets import load_iris

CONCURRENCY = 20        # Number of threads (concurrent requests)
TOTAL_REQUESTS = 1000     # Total number of requests to send
client = bentoml.SyncHTTPClient("http://localhost:3000")

iris = load_iris()
data_samples = iris.data.tolist()
payloads = [random.choice(data_samples) for _ in range(TOTAL_REQUESTS)]

def send_request(index, data):
    """Send a single HTTP request and print the result."""
    try:
        start_time = time.time()
        response = client.predict(np.array([data]))
        duration = time.time() - start_time
    except Exception as e:
        print(f"Request {index}: Error -> {e}")

print(f"Sending {TOTAL_REQUESTS} requests to {client.url} with concurrency {CONCURRENCY}...")
with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    for i, data in enumerate(payloads, start=1):
        executor.submit(send_request, i, data)

print("Done.")
