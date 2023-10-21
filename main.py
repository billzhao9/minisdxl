from typing import Any, Optional, Union,List,Type
import tempfile
import requests
import torch
import base64
from io import BytesIO
from PIL import Image
import time
import json
import functools
from math import ceil, sqrt
import io
import os
import re
import sys
from tqdm import tqdm
from pydantic import BaseModel
import gc
from lpw_stable_diffusion_xl import SDXLLongPromptWeightingPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers import (
    AutoencoderKL,
    ConfigMixin,
    LMSDiscreteScheduler,
    DDIMScheduler,
    ModelMixin,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    PretrainedConfig,
    PreTrainedModel,
)


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
    lora_scale: Optional[float] = 1.0


cache_dir = "/persistent-storage/"
init_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
HF_AUTH_TOKEN = "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"
device = "cuda" if torch.cuda.is_available() else "cpu"
lora_dir = "/persistent-storage/sdxl/lora/"
serialmodel_path1 = rf"{cache_dir}saved/sdxl/base/"
os.makedirs(serialmodel_path1, exist_ok=True)
prev_lora = None
prev_loras = []
prev_lora_scale = 1
is_model_changed = False
kept_lora_keyword = ""

from tensorizer.utils import no_init_or_tensor
from tensorizer import TensorDeserializer, TensorSerializer, stream_io, utils

def serialize_model(
    model: torch.nn.Module,
    config: Optional[Union[ConfigMixin, AutoConfig, dict]],
    model_directory: str,
    model_prefix: str = "model",
):
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays and serialize them to GooseTensor format. The stripped
    model is also serialized to pytorch format.

    Args:
        model: The model to serialize.
        config: The model's configuration. This is optional and only
            required for HuggingFace Transformers models. Diffusers
            models do not require this.
        model_directory: The directory to save the serialized model to.
        model_prefix: The prefix to use for the serialized model files. This
            is purely optional, and it allows for multiple models to be
            serialized to the same directory. A good example are Stable
            Diffusion models. Default is "model".
    """

    os.makedirs(model_directory, exist_ok=True)
    dir_prefix = f"{model_directory}/{model_prefix}"

    if config is None:
        config = model
    if config is not None:
        if hasattr(config, "to_json_file"):
            config.to_json_file(f"{dir_prefix}-config.json")
        if isinstance(config, dict):
            with open(f"{dir_prefix}-config.json", "w") as config_file:
                config_file.write(json.dumps(config, indent=2))

    ts = TensorSerializer(f"{dir_prefix}.tensors")
    start = time.time()
    ts.write_module(model)
    end = time.time()
    print((f"Serialising model took {end - start} seconds"),  file=sys.stderr)
    ts.close()

def load_model(
    path_uri: str,
    model_class: Union[
        Type[PreTrainedModel], Type[ModelMixin], Type[ConfigMixin]
    ],
    config_class: Optional[
        Union[Type[PretrainedConfig], Type[ConfigMixin], Type[AutoConfig]]
    ] = None,
    model_prefix: Optional[str] = "model",
    device: torch.device = utils.get_device(),
    dtype: Optional[str] = None,
) -> torch.nn.Module:
    """
    Given a path prefix, load the model with a custom extension

    Args:
        path_uri: path to the model. Can be a local path or a URI
        model_class: The model class to load the tensors into.
        config_class: The config class to load the model config into. This must be
            set if you are loading a model from HuggingFace Transformers.
        model_prefix: The prefix to use to distinguish between multiple serialized
            models. The default is "model".
        device: The device onto which to load the model.
        dtype: The dtype to load the tensors into. If None, the dtype is inferred from
            the model.
    """

    if model_prefix is None:
        model_prefix = "model"

    begin_load = time.time()
    ram_usage = utils.get_mem_usage()

    config_uri = f"{path_uri}/{model_prefix}-config.json"
    tensors_uri = f"{path_uri}/{model_prefix}.tensors"
    tensor_stream = stream_io.open_stream(tensors_uri)

    print(f"Loading {tensors_uri}, {ram_usage}")

    tensor_deserializer = TensorDeserializer(
        tensor_stream, device=device, dtype=dtype, lazy_load=True
    )

    if config_class is not None:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_config_path = os.path.join(temp_dir, "config.json")
                with open(temp_config_path, "wb") as temp_config:
                    temp_config.write(stream_io.open_stream(config_uri).read())
                config = config_class.from_pretrained(temp_dir)
                config.gradient_checkpointing = True
        except ValueError:
            config = config_class.from_pretrained(config_uri)
        with utils.no_init_or_tensor():
            # AutoModels instantiate from a config via their from_config()
            # method, while other classes can usually be instantiated directly.
            config_loader = getattr(model_class, "from_config", model_class)
            model = config_loader(config)
    else:
        try:
            config = json.loads(
                stream_io.open_stream(config_uri).read().decode("utf-8")
            )
        except ValueError:
            with open(config_uri, "r") as f:
                config = json.load(f)
        with utils.no_init_or_tensor():
            model = model_class(**config)

    tensor_deserializer.load_into_module(model)

    tensor_load_s = time.time() - begin_load
    rate_str = utils.convert_bytes(
        tensor_deserializer.total_bytes_read / tensor_load_s
    )
    tensors_sz = utils.convert_bytes(tensor_deserializer.total_bytes_read)
    print(
        f"Model tensors loaded in {tensor_load_s:0.2f}s, read "
        f"{tensors_sz} @ {rate_str}/s, {utils.get_mem_usage()}"
    )

    return model

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
    global prev_lora, is_model_changed,loggerg
    if prev_lora != new_lora_id:
        loggerg.info("lora ID has been changed, need to reload model")
        prev_lora = new_lora_id
        is_model_changed = True
        

def check_lora_ids(new_lora_ids):
    global prev_loras, is_model_changed,loggerg
    if prev_loras != new_lora_ids:
        loggerg.info("lora IDs has been changed, need to reload model")
        prev_loras = new_lora_ids
        is_model_changed = True

def check_lora_scale(new_lora_scale):
    global prev_lora_scale, is_model_changed,loggerg
    if prev_lora_scale != new_lora_scale:
        loggerg.info("lora Scale has been changed, need to reload model")
        prev_lora_scale = new_lora_scale
        is_model_changed = True

def init_model():
    print("init model")
    
    global base
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, use_auth_token=HF_AUTH_TOKEN,cache_dir=cache_dir
    )
    base.to(device)

    serialize_model(
            base.text_encoder.eval(),
            base.text_encoder.config,
            serialmodel_path1,
            "encoder",
        )
    
    serialize_model(
            base.text_encoder_2.eval(),
            base.text_encoder_2.config,
            serialmodel_path1,
            "encoder_2",
        )
    serialize_model(base.vae.eval(), None, serialmodel_path1, "vae")
    serialize_model(base.unet.eval(), None, serialmodel_path1, "unet")


global starttime

def start():
    global base
    base = None
    global starttime
    os.makedirs(lora_dir, exist_ok=True)
    starttime = time.time()
    filename = rf"{serialmodel_path1}vae.tensors"
    print(filename)
    if not os.path.exists(filename):
        with no_init_or_tensor():
                # Load your model here using whatever class you need to initialise an empty model from a config.
                init_model()
    end_init = time.time() - starttime

    print(f"Initialising default model took {end_init} seconds",  file=sys.stderr)
    if base is not None:
        del base
        gc.collect()


    #vae = load_model(serialmodel_path1, AutoencoderKL, None, "vae", device)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, cache_dir=cache_dir)
    unet = load_model(serialmodel_path1, UNet2DConditionModel, None, "unet", device)
    encoder = load_model(serialmodel_path1, CLIPTextModel, CLIPTextConfig, "encoder", device)
    encoder_2 = load_model(serialmodel_path1, CLIPTextModelWithProjection, CLIPTextConfig, "encoder_2", device)

    base = SDXLLongPromptWeightingPipeline(
            text_encoder = encoder,
            text_encoder_2 = encoder_2,
            vae=vae,
            unet=unet,
            tokenizer=CLIPTokenizer.from_pretrained(
                init_model_id,subfolder="tokenizer",cache_dir=cache_dir
            ),
            tokenizer_2=CLIPTokenizer.from_pretrained(
                init_model_id,subfolder="tokenizer_2",cache_dir=cache_dir
            ),
            scheduler=DDIMScheduler.from_pretrained(
                init_model_id,subfolder="scheduler",cache_dir=cache_dir
            ),
        ).to(device)
    global end_load
    end_load = time.time() - starttime

    print(f"loading the model took {end_load} seconds",  file=sys.stderr)
    base.enable_xformers_memory_efficient_attention()
    base.enable_attention_slicing()


start()

API_URL = "https://api.tzone03.xyz/"

def request_civitai_detail(url):
    pattern = r'https://civitai\.com/models/(.+)'
    m = re.match(pattern, url)
    if not m:
        return False, "不是一个有效的 civitai 模型页面链接，暂不支持"

    req_url = API_URL + "civitai/models/" + m.group(1)
    #res = requests.get(req_url)

    try:
        res = requests.get(req_url, timeout=10)  # 设置超时时间为10秒
    except requests.exceptions.Timeout:
        return False, "请求超时，请稍后再试"
    except requests.exceptions.RequestException as e:
        return False, f"请求失败: {e}"

    if res.status_code >= 500:
        return False, "呃 服务好像挂了，理论上我应该在修了，可以进群看看进度……"
    if res.status_code >= 400:
        return False, "不是一个有效的 civitai 模型页面链接，暂不支持"

    if res.ok:
        return True, res.json()
    else:
        return False, res.text
    

def check_lora_info(url):
    global loggerg

    loggerg.info("check lora info")
    ok, resp = request_civitai_detail(url)
    if not ok:
        loggerg.info("return empty lora info")
        loggerg.info(rf"error: {resp}")
        return "empty","empty","empty"

    has_download_file = False
    if resp["version"]["file"]["downloadUrl"]:
        has_download_file = True
        loggerg.info(resp["version"]["file"]["name"])
        loggerg.info(resp["version"]["file"]["downloadUrl"])
        loggerg.info(resp["version"]["trainedWords"])
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
    lora_scale = item.lora_scale

    global is_model_changed,kept_lora_keyword,loggerg
    global end_load
    loggerg = logger
    global base,starttime
    starttime = time.time()
    lora_name = ""
    lora_url = ""
    lora_keyword = []
    logger.info("inference")
    # txt2img starts
    if command == "txt2img":
        check_lora_id(lora)
        check_lora_ids(loras)
        check_lora_scale(lora_scale)
            


        if is_model_changed:
            base.unfuse_lora()
            if len(lora) < 1:
                base.unload_lora_weights()
            if len(loras) < 1:
                base.unload_lora_weights()
                
            if len(lora) >= 1:
                if lora[0].startswith("http://") or lora[0].startswith("https://"):
                    logger.info("download remote lora")
                    lora_name, lora_url, lora_keyword = check_lora_info(lora[0])
                    if lora_name != "empty":
                        lora_path = f"{lora_dir}{lora_name}"
                        download_model(lora_url,lora_dir,lora_name)
                        if os.path.exists(lora_path):
                            base.unload_lora_weights()
                            kept_lora_keyword = ""
                            base.load_lora_weights(pretrained_model_name_or_path_or_dict =lora_path,weight_name= lora_name)
                            logger.info(f"Loaded local downloaded LoRA weights!  {lora_name}")
                            if len(lora_keyword) > 0:
                                if kept_lora_keyword== "":
                                    kept_lora_keyword =  lora_keyword[0]
                                else:
                                    kept_lora_keyword = kept_lora_keyword + "," + lora_keyword[0]
                                prompt = prompt + "," + lora_keyword[0]
                        else:
                            logger.info("can not find out the lora")
                    else:
                          logger.info("can not find out the lora")
                else:
                    logger.info("download hf lora")

                    # load single lora
                    if len(lora) == 2:
                        logger.info("unload previous lora")
                        base.unload_lora_weights()
                        kept_lora_keyword = ""
                        logger.info("load single lora")
                        repo_id = lora[0]
                        weight_name = lora[1]
                        base.load_lora_weights(pretrained_model_name_or_path_or_dict =repo_id,weight_name= weight_name, cache_dir=cache_dir)
                        logger.info("Loaded LoRA weights!")
                    else:
                        logger.info("lora pair information is not correct")

            # load multiple lora
            if bool(loras):
                logger.info("unload previous lora")
                base.unload_lora_weights()
                kept_lora_keyword = ""
                logger.info('load multiple lora')
                for sub_lora in loras:
                    if len(sub_lora) > 1:
                        logger.info("load HF lora")
                        sub_lora_repo_id = sub_lora[0]
                        sub_lora_weight_name = sub_lora[1]
                        base.load_lora_weights(pretrained_model_name_or_path_or_dict =sub_lora_repo_id,weight_name= sub_lora_weight_name, cache_dir=cache_dir)
                        logger.info(f"loaded lora weight {sub_lora_weight_name}")
                    else:
                        if sub_lora[0].startswith("http://") or sub_lora[0].startswith("https://"):
                            logger.info("download remote lora")
                            lora_name, lora_url, lora_keyword = check_lora_info(sub_lora[0])
                            if lora_name != "empty":
                                lora_path = f"{lora_dir}{lora_name}"
                                download_model(lora_url,lora_dir,lora_name)
                                if os.path.exists(lora_path):
                                    base.load_lora_weights(pretrained_model_name_or_path_or_dict =lora_path,weight_name= lora_name)
                                    logger.info(f"Loaded multiple local downloaded LoRA weights!  {lora_name}")
                                    if len(lora_keyword) > 0:
                                        if kept_lora_keyword== "":
                                            kept_lora_keyword =  lora_keyword[0]
                                        else:
                                            kept_lora_keyword = kept_lora_keyword + "," + lora_keyword[0]

                                        prompt = prompt + "," + lora_keyword[0]
                                else:
                                    logger.info("can not find out the lora")
                            else:
                                 logger.info("can not find out the lora")

            else:
                logger.info("no multiple loras")

        else:
            logger.info("Using previously loaded LoRA weights.")
            prompt = prompt + "," + kept_lora_keyword
        
        base.fuse_lora(lora_scale=lora_scale)

        is_model_changed = False
        generator = None
        if seed:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        else:
            seed = torch.randint(0, 1000000, (1,)).item()
            generator = torch.Generator(device=device).manual_seed(int(seed))
        
        if negative_prompt is None:
            negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"

        images = []
        finished_images = []
        logger.info(rf"prompt: {prompt}")
        with torch.inference_mode():
            images = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                output_type="latent" if use_refiner else "pil",
                target_size=(width, height)
            ).images


            if images is not None:
                logger.info("finalizing return images")        
                if grid == True:
                    gridimg = image_grid(images)
                    images.insert(0, gridimg)
                    logger.info("append a grid img")

                for image in images:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

                #pdb.set_trace()
                end_return = time.time() - starttime
                logger.info(f"The infrence process took {end_return} seconds")
                logger.info(f"The model loading process took {end_load} seconds")
                return finished_images

            else:
                return {"result": False}
            



