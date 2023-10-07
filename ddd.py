from PIL import Image
import numpy as np
import requests
import os
import re
import subprocess
import threading

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

def resp_to_components(resp):
    if resp == None:
        return [None, None, None, None, None, None, None, None, None, None]

    img = resp["version"]["image"]["url"]
    #if img:
        #img = process_image(img)

    return [
        resp["name"],
        resp["type"],
        ", ".join(resp["version"]["trainedWords"]),
        resp["creator"]["username"],
        ", ".join(resp["tags"]),
        resp["version"]["updatedAt"],
        resp["description"],
        img,
        resp["version"]["file"]["name"],
        resp["version"]["file"]["downloadUrl"],
    ]

def preview(url):
    ok, resp = request_civitai_detail(url)
    if not ok:
        return [resp]

    has_download_file = False
    more_guides = ""
    if resp["version"]["file"]["downloadUrl"]:
        has_download_file = True
        print(resp["version"]["file"]["name"])
        print(resp["version"]["file"]["downloadUrl"])
        print(resp["version"]["trainedWords"])
        #print(resp["tags"])

preview("https://civitai.com/models/124016/sdxl-chalk-dust-style")