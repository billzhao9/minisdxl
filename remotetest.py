import requests
import json
import base64
from io import BytesIO
from PIL import Image
import datetime
now = datetime.datetime.now()

url = rf"https://run.cerebrium.ai/v3/p-0159d6c7/minitxt2imgxl/predict"

headers = {
    "Authorization": "public-09a6fceaeb413fa97e83",
    "Content-Type": "application/json"
}

# url = rf"https://run.cerebrium.ai/v2/p-3de95df7/txt2img/predict"

# headers = {
#     "Authorization": "public-f0fa3e6af1a5c7915b30",
#     "Content-Type": "application/json"
# }


data = {
    "command": "txt2img",
    "prompt": "a panda",
    "num_images_per_prompt": 1,
    "num_inference_steps": 15,
    "guidance_scale": 9,
    #"loras": [["https://civitai.com/models/145479/three-starmound"]],
    "grid": False
}
date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())
# with open('output.txt', 'w+') as f:
#     f.write(str(response.json()))


for i, img_str in enumerate(response.json()["result"]):
    # decode the base64-encoded string to bytes
    img_data = base64.b64decode(img_str)
    # convert the bytes to a PIL image object
    img = Image.open(BytesIO(img_data))
    # save the image to disk with a unique filename
    filename = f"images/image_{i}_{date_string}.png"
    with open(filename, "wb") as f:
        img.save(f, format="PNG")
