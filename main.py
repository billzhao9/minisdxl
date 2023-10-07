from typing import Any, Optional, Union,List
import requests
from diffusers import DiffusionPipeline, AutoencoderKL,UNet2DConditionModel
import torch
import base64
from io import BytesIO
from PIL import Image
import time
import functools
from math import ceil, sqrt
import io
import os
import re
from tqdm import tqdm
from pydantic import BaseModel


class Item(BaseModel):
    command: Optional[str] = ""
    images_data: Optional[Union[str, List[str]]] = []
    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt:Optional[Union[str, List[str]]] = None
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    num_inference_steps: Optional[int] = 25
    num_images_per_prompt: Optional[int] = 1
    seed: Optional[int] = None
    guidance_scale: Optional[int] = 9
    grid: Optional[bool] = False
    lora: Optional[List[str]] = []
    loras: Optional[List[List[str]]] = []
    use_refiner: Optional[bool] = False
    high_noise_frac: Optional[float] = 0.8


cache_dir = "/persistent-storage/"
init_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
HF_AUTH_TOKEN = "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"
device = "cuda" if torch.cuda.is_available() else "cpu"
lora_dir = "/persistent-storage/sdxl/lora/"

prev_lora = None
prev_loras = []
is_model_changed = False
kept_lora_keyword = ""

def download_model(model_url, cache_dir, filename):
    # 确保 cache_dir 以 '/' 结尾，以便正确构建完整的文件路径
    cache_dir = os.path.join(cache_dir, '')
    # 如果 cache_dir 目录不存在，则创建它
    os.makedirs(cache_dir, exist_ok=True)
    # 构建完整的文件路径
    file_path = f"{cache_dir}{filename}"
    # 获取文件大小，以便在进度条中显示
    expected_file_size = int(requests.head(model_url).headers.get('content-length', 0))
    # 检查文件是否已存在，如果不存在则开始下载
    if not os.path.exists(file_path):
        print(f'Downloading {filename}...')
        print("Download Location:", file_path)
        response = requests.get(model_url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        # 使用 'with' 语句确保文件和进度条在完成后正确关闭
        with open(file_path, 'wb') as f, tqdm(desc="Downloading", unit="bytes", total=expected_file_size) as progress:
            # 分块读取和写入文件，同时更新进度条
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # 过滤掉空块，以防止写入空数据
                    f.write(chunk)
                    progress.update(len(chunk))
        print(f'Download {filename} complete')
    else:
        print(f'{filename} already exists and is complete.')


def convert_to_b64(self, image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_b64

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


def check_lora_id(new_lora_id):
    global prev_lora, is_model_changed
    if prev_lora != new_lora_id:
        print("lora ID has been changed, need to reload model")
        prev_lora = new_lora_id
        is_model_changed = True
        

def check_lora_ids(new_lora_ids):
    global prev_loras, is_model_changed
    if prev_loras != new_lora_ids:
        print("lora IDs has been changed, need to reload model")
        prev_loras = new_lora_ids
        is_model_changed = True

def init_model():
    print("init model")
    os.makedirs(lora_dir, exist_ok=True)

    global pipe, refiner
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, cache_dir=cache_dir)
    #unet = UNet2DConditionModel.from_pretrained(init_model_id, subfolder="unet", in_channels=9, low_cpu_mem_usage=False, ignore_mismatched_sizes=True,cache_dir=cache_dir)
    pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=False,
                cache_dir=cache_dir,
                #unet=unet,
                low_cpu_mem_usage=False, 
            )
    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()

    refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipe.text_encoder_2,
            vae=pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=cache_dir,
        )
    
    refiner.to(device)
    refiner.enable_xformers_memory_efficient_attention()


init_model()




API_URL = "https://api.tzone03.xyz/"

def request_civitai_detail(url):
    pattern = r'https://civitai\.com/models/(.+)'
    m = re.match(pattern, url)
    if not m:
        return False, "不是一个有效的 civitai 模型页面链接，暂不支持"

    req_url = API_URL + "civitai/models/" + m.group(1)
    res = requests.get(req_url)

    if res.status_code >= 500:
        return False, "呃 服务好像挂了，理论上我应该在修了，可以进群看看进度……"
    if res.status_code >= 400:
        return False, "不是一个有效的 civitai 模型页面链接，暂不支持"

    if res.ok:
        return True, res.json()
    else:
        return False, res.text
    

def check_lora_info(url):
    print("check lora info")
    ok, resp = request_civitai_detail(url)
    if not ok:
        return [resp]

    has_download_file = False
    if resp["version"]["file"]["downloadUrl"]:
        has_download_file = True
        print(resp["version"]["file"]["name"])
        print(resp["version"]["file"]["downloadUrl"])
        print(resp["version"]["trainedWords"])
        return resp["version"]["file"]["name"],resp["version"]["file"]["downloadUrl"],resp["version"]["trainedWords"]

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
    lora = item.lora
    loras = item.loras
    use_refiner = item.use_refiner
    high_noise_frac = item.high_noise_frac
    

    global pipe, refiner,is_model_changed,kept_lora_keyword
    lora_name = ""
    lora_url = ""
    lora_keyword = []

    # txt2img starts
    if command == "txt2img":
        check_lora_id(lora)
        check_lora_ids(loras)



        if is_model_changed:

            if len(lora) < 1:
                pipe.unload_lora_weights()
            if len(loras) < 1:
                pipe.unload_lora_weights()
                
            if len(lora) >= 1:
                if lora[0].startswith("http://") or lora[0].startswith("https://"):
                    print("download remote lora")
                    lora_name, lora_url, lora_keyword = check_lora_info(lora[0])
                    lora_path = f"{lora_dir}{lora_name}"
                    download_model(lora_url,lora_dir,lora_name)
                    if os.path.exists(lora_path):
                        pipe.unload_lora_weights()
                        kept_lora_keyword = ""
                        pipe.load_lora_weights(pretrained_model_name_or_path_or_dict =lora_path,weight_name= lora_name)
                        print(f"Loaded local downloaded LoRA weights!  {lora_name}")
                        if len(lora_keyword) > 0:
                            if kept_lora_keyword== "":
                                kept_lora_keyword =  lora_keyword[0]
                            else:
                                kept_lora_keyword = kept_lora_keyword + "," + lora_keyword[0]
                            prompt = prompt + "," + lora_keyword[0]
                    else:
                         print("can not find out the lora")
                else:
                    print("download hf lora")

                    # load single lora
                    if len(lora) == 2:
                        print("unload previous lora")
                        pipe.unload_lora_weights()
                        kept_lora_keyword = ""
                        print("load single lora")
                        repo_id = lora[0]
                        weight_name = lora[1]
                        pipe.load_lora_weights(pretrained_model_name_or_path_or_dict =repo_id,weight_name= weight_name, cache_dir=cache_dir)
                        print("Loaded LoRA weights!")
                    else:
                        print("lora pair information is not correct")
                

            # load multiple lora
            if bool(loras):
                print("unload previous lora")
                pipe.unload_lora_weights()
                kept_lora_keyword = ""
                print('load multiple lora')
                for sub_lora in loras:
                    if len(sub_lora) > 1:
                        print("load HF lora")
                        sub_lora_repo_id = sub_lora[0]
                        sub_lora_weight_name = sub_lora[1]
                        pipe.load_lora_weights(pretrained_model_name_or_path_or_dict =sub_lora_repo_id,weight_name= sub_lora_weight_name, cache_dir=cache_dir)
                        print(f"loaded lora weight {sub_lora_weight_name}")
                    else:
                        if sub_lora[0].startswith("http://") or sub_lora[0].startswith("https://"):
                            print("download remote lora")
                            lora_name, lora_url, lora_keyword = check_lora_info(sub_lora[0])
                            lora_path = f"{lora_dir}{lora_name}"
                            download_model(lora_url,lora_dir,lora_name)
                            if os.path.exists(lora_path):
                                pipe.load_lora_weights(pretrained_model_name_or_path_or_dict =lora_path,weight_name= lora_name)
                                print(f"Loaded multiple local downloaded LoRA weights!  {lora_name}")
                                if len(lora_keyword) > 0:
                                    if kept_lora_keyword== "":
                                        kept_lora_keyword =  lora_keyword[0]
                                    else:
                                        kept_lora_keyword = kept_lora_keyword + "," + lora_keyword[0]

                                    prompt = prompt + "," + lora_keyword[0]
                            else:
                                print("can not find out the lora")

            else:
                print("no multiple loras")

        else:
            print("Using previously loaded LoRA weights.")
            prompt = prompt + "," + kept_lora_keyword

        is_model_changed = False
        generator = None
        if seed:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        else:
            seed = torch.randint(0, 1000000, (1,)).item()
            generator = torch.Generator(device=device).manual_seed(int(seed))

        images = []
        finished_images = []
        print(rf"prompt: {prompt}")
        with torch.inference_mode():
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                denoising_end=high_noise_frac if use_refiner else 1.0,
                output_type="latent" if use_refiner else "pil",
                target_size=(width, height)
            ).images


            if use_refiner:
                images = refiner(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    denoising_start=high_noise_frac,
                    image=images[None, :],
                    target_size=(width, height)
                ).images

            if images is not None:
                print("finalizing return images")        
                if grid == True:
                    gridimg = image_grid(images)
                    images.insert(0, gridimg)
                    logger.info("append a grid img")

                for image in images:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

                #pdb.set_trace()
                return finished_images

            else:
                return {"result": False}
            



