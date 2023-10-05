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
data = {
    "command": "txt2img",
    #prompt": "一只带着红帽子的大老虎, 一直带着粉色帽子的大老虎，一直带着绿的帽子的大老虎，一直带着蓝色帽子的兔子",
    #"prompt": "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    #"prompt": "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k",
    "prompt":"Generate a vivid and captivating image of a super-powered dog with extraordinary abilities, showcasing its heroic stature, sleek physique, and dynamic energy in an awe-inspiring setting.",
    #"prompt":["very complex hyper-maximalist overdetailed cinematic tribal darkfantasy closeup portrait of a malignant beautiful young dragon queen goddess megan fox with long black windblown hair and dragon scale wings, Magic the gathering, pale skin and dark eyes,flirting smiling succubus confident seductive, gothic, windblown hair, vibrant high contrast, by andrei riabovitchev, tomasz alen kopera,moleksandra shchaslyva, peter mohrbacher, Omnious intricate, octane, moebius, arney freytag, Fashion photo shoot, glamorous pose, trending on ArtStation, dramatic lighting, ice, fire and smoke, orthodox symbolism Diesel punk, mist, ambient occlusion, volumetric lighting, Lord of the rings, BioShock, glamorous, emotional, tattoos,shot in the photo studio, professional studio lighting, backlit, rim lightingDeviant-art, hyper detailed illustration, 8k" , "photo of a gorgeous young woman in the style of stefan kostic and david la chapelle, coy, shy, alluring, evocative, stunning, award winning, realistic, sharp focus, 8 k high definition, 3 5 mm film photography, photo realistic, insanely detailed, intricate, elegant, art by stanley lau and artgerm", "a portrait of a cute girl with a luminous dress, eyes shut, mouth closed, long hair, wind, sky, clouds, the moon, moonlight, stars, universe, fireflies, butterflies, lights, lens flares effects, swirly bokeh, brush effect, In style of Yoji Shinkawa, Jackson Pollock, wojtek fus, by Makoto Shinkai, concept art, celestial, amazing, astonishing, wonderful, beautiful, highly detailed, centered", "a beautiful Cotton Mill Girl, symmetrical, centered, dramatic angle, ornate, details, smooth, sharp focus, illustration, realistic, cinematic, artstation, award winning, rgb , unreal engine, octane render, cinematic light, macro, depth of field, blur, red light and clouds from the back, highly detailed epic cinematic concept art CG render made in Maya, Blender and Photoshop, octane render, excellent composition, dynamic dramatic cinematic lighting, aesthetic, very inspirational, arthouse by Henri Cartier Bresson"],
    #"prompt": ['a photograph of an astronaut riding a horse','A fluffy pomeranian that is equally black and white fur','a girl in a red dress, sunset, beautiful face','a beautiful tiger'],
    #"prompt": "chinese water painting, moxin, A fluffy pomeranian that is equally black and white fur, looking at an iphone, happy, tongue sticking out, silly, drawn in the french Tintin et Milou comic style. --s 250 ",
    #"prompts": ['a photograph of an astronaut riding a horse','A fluffy pomeranian that is equally black and white fur','a girl in a red dress, sunset, beautiful face','a beautiful tiger'],
    #"prompts": ['一只粉色的狗狗', ' 一只红色的虎猫', '一只绿的猴子' ,'一只带着蓝色帽子的兔子'],
    #"negative_prompt": 'ugly, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, poor lighting, dull, and unclear.', 
    # "negative_prompts": ['ugly, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, poor lighting, dull, and unclear.', 
    #                      'ugly, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, poor lighting, dull, and unclear.', 
    #                      'ugly, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, poor lighting, dull, and unclear.' ,
    #                      'ugly, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, poor lighting, dull, and unclear.'],
    #"height": 768,
    #"width": 768,
    "num_images_per_prompt": 1,
    "num_inference_steps": 15,
    "guidance_scale": 9,
    #"seed": -1,
    "grid": False,
    #"model_id": "stablediffusionapi/deliberate-v2",
    #"model_id":r"sd_models/Base/deliberate_v2",
    #"use_custom_online_model": False,
    #"vae_id": "stabilityai/sd-vae-ft-mse",
    #"lora_id": "MoXinV1.safetensors",
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
