import base64
import io
import json
import os
import time
import urllib.request
import requests
import sys
from pathlib import Path
from typing import Optional, Union,List
from PIL import Image
from io import BytesIO
import numpy as np
import re
from pydantic import BaseModel
from math import ceil, sqrt

import torch
from diffusers import StableDiffusionXLPipeline,DiffusionPipeline
from diffusers import EulerDiscreteScheduler,DPMSolverMultistepScheduler,UniPCMultistepScheduler,PNDMScheduler

class Item(BaseModel):
    command: Optional[str]
    images_data: Optional[Union[str, List[str]]] = []
    model_id: Optional[str] = "stabilityai/stable-diffusion-xl-base-1.0"
    # prompt params
    prompt: Union[str, List[str]]
    negative_prompt:Optional[Union[str, List[str]]]
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    num_inference_steps: Optional[int] = 25
    num_images_per_prompt: Optional[int] = 1
    seed: Optional[int]
    guidance_scale: Optional[int] = 9
    grid: Optional[bool] = False

cache_dir = "/persistent-storage/"
init_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
HF_AUTH_TOKEN = "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"
device = "cuda" if torch.cuda.is_available() else "cpu"


def download_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

def match_lengths(images, prompts):
    image_length = len(images)
    prompt_length = len(prompts)
    
    if image_length != prompt_length:
        if image_length > prompt_length:
            # 如果prompts为空，使用一个默认值，否则复制最后一个元素
            value_to_append = 'same images' if not prompts else prompts[-1]
            prompts.extend([value_to_append] * (image_length - prompt_length))
        else:
            # 如果prompts的长度比images长，删除多余的prompts
            prompts = prompts[:image_length]

def image_grid(imgs):
    n = len(imgs)
    rows = int(sqrt(n))
    cols = ceil(n / rows)

    if rows > cols:
        rows, cols = cols, rows

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid

def init_model():
    print("init model")
    # global model
    # model = None
    # model = StableDiffusionXLPipeline.from_pretrained(init_model_id,torch_dtype=torch.float16,variant="fp16",use_auth_token=HF_AUTH_TOKEN,cache_dir=cache_dir,use_safetensors=True,requires_safety_checker=False).to(device)

    # model.enable_xformers_memory_efficient_attention()
    # model.enable_attention_slicing()
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16, use_safetensors=True)

    global base
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, use_auth_token=HF_AUTH_TOKEN,cache_dir=cache_dir
    )
    base.to(device)
    base.enable_xformers_memory_efficient_attention()
    base.enable_attention_slicing()

    global refiner
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        use_auth_token=HF_AUTH_TOKEN,
        cache_dir=cache_dir
    )
    refiner.to(device)


init_model()

def predict(item, run_id, logger, binaries=None):
    item = Item(**item)
    command = item.command
    images_data = item.images_data  
    #prompt related 
    prompt = item.prompt
    negative_prompt = item.negative_prompt
    guidance_scale = item.guidance_scale
    seed = item.seed
    num_inference_steps = item.num_inference_steps
    num_images_per_prompt = item.num_images_per_prompt
    width = item.width
    height = item.height
    grid = item.grid

    global model

    final_images = []
    if images_data:
        if isinstance(images_data, str):
            print("received image data")
            if images_data.startswith("http://") or images_data.startswith("https://"):
                image = download_image_from_url(images_data)
            else:
                image = Image.open(BytesIO(base64.b64decode(images_data))).convert("RGB")
            
            final_images.append(image)

        elif isinstance(images_data, list) and all(isinstance(item, str) for item in images_data):
            print("received image data array")
            temp_images = []
            for temp_image_data in images_data:
                if temp_image_data.startswith("http://") or temp_image_data.startswith("https://"):
                    temp_image = download_image_from_url(temp_image_data)
                else:       
                    temp_image = Image.open(BytesIO(base64.b64decode(temp_image_data))).convert("RGB")
                temp_images.append(temp_image)

            final_images = temp_images
    else:
        print("no image")


        prompts =[]
        negative_prompts =[]

        if prompt:
            if isinstance(prompt, str):
                prompts.append(prompt)
            elif isinstance(prompt, list) and all(isinstance(item, str) for item in prompt): 
                prompts = prompt

        if negative_prompt:
            if isinstance(negative_prompt, str):
                negative_prompts.append(negative_prompt)
            elif isinstance(negative_prompt, list) and all(isinstance(item, str) for item in negative_prompt): 
                negative_prompts = negative_prompt
        else:
            negative_prompts = ["canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render"]

        if images_data:
            negative_prompts = match_lengths(prompts, negative_prompts)
            prompts = match_lengths(final_images, prompts)
        else:
            negative_prompts = match_lengths(final_images, negative_prompts)

        prompt_embeds = None
        negative_prompt_embeds = None

        generator = None
        if seed:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        else:
            seed = torch.randint(0, 1000000, (1,)).item()
            generator = torch.Generator(device=device).manual_seed(int(seed))

    if command == "txt2img":
        global base, refiner
        images = []
        image = []
        with torch.inference_mode():
            # images = model(
            #     prompt=prompts,
            #     negative_prompt=negative_prompts,
            #     prompt_embeds=prompt_embeds, 
            #     negative_prompt_embeds=negative_prompt_embeds,
            #     height=height,
            #     width=width,
            #     guidance_scale=guidance_scale,
            #     num_images_per_prompt=num_images_per_prompt,
            #     num_inference_steps=num_inference_steps,
            #     generator=generator,
            # ).images
            #num_inference_steps = 40
            high_noise_frac = 0.8
            image = base(
                prompt=prompt,
                negative_prompt=negative_prompts,
                num_inference_steps=num_inference_steps,
                denoising_end=high_noise_frac,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                output_type="latent",
            ).images

            images = refiner(
                prompt=prompt,
                negative_prompt=negative_prompts,
                num_inference_steps=num_inference_steps,
                denoising_start=high_noise_frac,
                image=image,
            ).images

            if images is not None:
                print("finalizing return images")        
                if grid == True:
                    gridimg = image_grid(images)
                    images.insert(0, gridimg)
                    print("append a grid img")

                finished_images = []
                for image in images:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

                #pdb.set_trace()
                return finished_images

            else:
                return {"result": False}
            
    elif command == "img2img":
        print("img2img")
        pipe_i2i = AutoPipelineForImageToImage.from_pipe(model)
        images = None
        with torch.inference_mode():
            images = pipe_i2i(
                prompt=prompts,
                negative_prompt=negative_prompts,
                prompt_embeds=prompt_embeds, 
                negative_prompt_embeds=negative_prompt_embeds,
                image=final_images,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images

            if images is not None:
                print("finalizing return images")        
                if grid == True:
                    gridimg = image_grid(images)
                    images.insert(0, gridimg)
                    print("append a grid img")

                finished_images = []
                for image in images:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

                #pdb.set_trace()
                return finished_images

            else:
                return {"result": False}