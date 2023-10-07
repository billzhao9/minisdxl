import requests
import json
import base64


url = rf"https://rest-api.cerebrium.ai/update-model-scaling"

headers = {
    "Authorization": "private-5dbf4638f90afb58cfd5",
    "Content-Type": "application/json"
}

data = {
    "name": "minitxt2imgxl",
    "minReplicaCount": 0,
    "maxReplicaCount": 5,
    "cooldownPeriodSeconds": 180
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())