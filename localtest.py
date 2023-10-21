import requests
import json
import base64
from io import BytesIO
from PIL import Image
import datetime
now = datetime.datetime.now()

url = rf"http://127.0.0.1:8000/predict"
headers = {
    "Content-Type": "application/json"
}

#https://civitai.com/models/118845/jasmine-art-style-sdxl-10 #not working
#https://civitai.com/models/120126/sdxl-food-icons


data = {
    "command": "txt2img",
    #prompt": "一只带着红帽子的大老虎, 一直带着粉色帽子的大老虎，一直带着绿的帽子的大老虎，一直带着蓝色帽子的兔子",
    #"prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    #"prompt": "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k",
    #"prompt":"Generate a vivid and captivating image of a super-powered dog with extraordinary abilities, showcasing its heroic stature, sleek physique, and dynamic energy in an awe-inspiring setting.",
    "prompt": "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k,ikea",
    #"prompt": "a comfortable chair with a tv set",
    "num_images_per_prompt": 1,
    "num_inference_steps": 15,
    "guidance_scale": 9,
    #"seed": -1,
    "grid": False,
    #"lora": ["nerijs/pixel-art-xl","pixel-art-xl.safetensors"],
    #"lora": ["https://civitai.com/models/124016/sdxl-chalk-dust-style"],
    #"lora": ["ostris/ikea-instructions-lora-sdxl","ikea_instructions_xl_v1_5.safetensors"],
    #"loras": [["nerijs/pixel-art-xl","pixel-art-xl.safetensors"],["ostris/ikea-instructions-lora-sdxl","ikea_instructions_xl_v1_5.safetensors"]],
    #"loras": [["https://civitai.com/models/124016/sdxl-chalk-dust-style"]],
    #"loras": [["https://civitai.com/models/119157/sdxl-dragon-style"],["https://civitai.com/models/124016/sdxl-chalk-dust-style"]],
    #"loras": [["https://civitai.com/models/118992/salvador-dali-sdxl-10-art-style-lora"]],
    "loras": [["https://civitai.com/models/145479/three-starmound"]],
    "lora_scale":0.9,
    #"embedding_id": "style-empire.pt"
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
