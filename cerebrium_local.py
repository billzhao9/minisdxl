# This script allows to simulate Cerebrium serverless in local for test purpose
# Author: @chaignc
# run me with
# uvicorn cerebrium_local:app
# uvicorn cerebrium_local:app --reload
# cerebrium deploy --hardware A10 --api-key private-41fdf046e5a1d00801e8 txt2img

# cerebrium deploy --hardware AMPERE_A5000 --api-key private-5dbf4638f90afb58cfd5 minitxt2imgxl

from fastapi import FastAPI
from mainlocal import predict
import logging
import time

app = FastAPI()
logger = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Define the predict endpoint
@app.post("/predict")
def predict_route(input_data: dict):
    start_time = time.time()
    result = predict(input_data, "local", logger, binaries=None)
    end_time = time.time()
    run_time_ms = round((end_time - start_time) * 1000, 2)
    return {"result": result, "run_time_ms": run_time_ms}