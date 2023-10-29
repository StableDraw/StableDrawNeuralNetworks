from typing import List
import io
from PIL import Image

from webui_api import web_api

def webui_text2img(caption: str, params: dict) -> bytes:
    '''
    params={
        "host": "127.0.0.1", #хост где запущен webui
        "port": 7860, #порт где запущен webui
        "save": False, #сохранять картинки после генерации в папку save_dir
        "save_dir": "", #куда сохранять
        "enable_hr": False, #использовать апскейлер после генерации
        "denoising_strength": 0.7, #(от 0.0 до 1.0)
        "hr_scale": 2, #множитель апскейла
        "hr_upscaler": "Latent", #вид апскейлера (Latent, LatentAntialiased, LatentBicubic, LatentBicubicAntialiased, LatentNearest, LatentNearestExact, Lanczos, Nearest, ESRGAN_4x, LDSR, ScuNET_GAN, ScuNET_PSNR, SwinIR_4x)
        "seed": -1, # ядро генерации (-1 - случайное; [0, 1000000000])
        "sampler_name": "Euler a", #обработчик (Euler a, Euler, DPM2, DPM2 a, DDIM, ...)
        "batch_size": 1, #размер батча
        "steps": 20, #кол-во шагов генерации
        "cfg_scale": 7.0, #cfg (от 0.0 до 30.0)
        "width": 512, #ширина генерируемого изображения
        "height": 512, #высота генерируемого изображения
        "restore_faces": False, #делает более реалистичные лица
        "tiling": False, #делает изображение, все грани которого не содержат каких-то четких элементов.
        "negative_prompt": "", #антипромт
        "eta":1.0, #Только для обработчика "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
        "s_churn":0, #Только для обработчиков "EulerEDMSampler" или "HeunEDMSampler" (от 0.0 до 1.0)
        "s_tmax":0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы меньше этого значения (от 0.0 до "sigma_max" и < "s_tmax")
        "s_tmin":0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы больше этого значения (от "sigma_min" до "sigma_max" и > "s_tmin")
        "s_noise":1, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler" или "EulerAncestralSampler" или "DPMPP2SAncestralSampler") и "s_churn" > 0 (от 0.0)
    }
    '''
    result = web_api.text2img(prompt=caption, **params)
    img_byte_arr = io.BytesIO()
    result.image.save(img_byte_arr, format='PNG')
    binary_data = img_byte_arr.getvalue()
    return binary_data

def webui_text2img_controlnet(init_img_binary_data: bytes, caption: str, params: dict) -> List[bytes]:
    '''
    params={
        "host": "127.0.0.1", #хост где запущен webui
        "port": 7860, #порт где запущен webui
        "save": False, #сохранять картинки после генерации в папку save_dir
        "save_dir": "", #куда сохранять
        "negative_prompt": "", #антипромпт
        "mask_image": None, #маска для inpainting
        "module": "None", #препроцессор
        "model": "None", #модель
        "weight": 1.0, #при низком получаются более размытые изображения (от 0.0 до 30.0)
        "resize_mode": str = "Resize and Fill", #(Resize and Fill, Crop and Resize, Just Resize)
        "control_mode": int = 0, #(0 - Баланс, 1 - Мой промт важнее, 2 - контролнет важнее)
        "pixel_perfect": bool = False, #делает изображение более четким
    }
    '''
    image = Image.open(io.BytesIO(init_img_binary_data))
    binary_data_list = []
    result = web_api.text2img_controlnet(prompt=caption, input_image=image, **params)
    for img in result.images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        binary_data_list.append(img_byte_arr.getvalue())
    return binary_data_list

def webui_img2img(init_img_binary_data_list: List[bytes], caption: str, params: dict) -> bytes:
    '''
    params={
        "host": "127.0.0.1", #хост где запущен webui
        "port": 7860, #порт где запущен webui
        "save": False, #сохранять картинки после генерации в папку save_dir
        "save_dir": "", #куда сохранять
        "denoising_strength": 0.7, #(от 0.0 до 1.0)
        "image_cfg_scale": 1.5, #cfg для картинки (от 0.0 до 30.0)
        "mask_image": None,  # PIL Image mask для inpainting
        "mask_blur": 4,
        "inpainting_fill": 0,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 0,
        "inpainting_mask_invert": 0,
        "initial_noise_multiplier": 1,
        "seed": -1, # ядро генерации (-1 - случайное; [0, 1000000000])
        "sampler_name": "Euler a", #обработчик (Euler a, Euler, DPM2, DPM2 a, DDIM, ...)
        "batch_size": 1, #размер батча
        "steps": 20, #кол-во шагов генерации
        "cfg_scale": 7.0, #cfg (от 0.0 до 30.0)
        "width": 512, #ширина генерируемого изображения
        "height": 512, #высота генерируемого изображения
        "restore_faces": False, #делает более реалистичные лица
        "tiling": False, #делает изображение, все грани которого не содержат каких-то четких элементов.
        "negative_prompt": "", #антипромт
        "eta":1.0, #Только для обработчика "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
        "s_churn":0, #Только для обработчиков "EulerEDMSampler" или "HeunEDMSampler" (от 0.0 до 1.0)
        "s_tmax":0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы меньше этого значения (от 0.0 до "sigma_max" и < "s_tmax")
        "s_tmin":0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы больше этого значения (от "sigma_min" до "sigma_max" и > "s_tmin")
        "s_noise":1, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler" или "EulerAncestralSampler" или "DPMPP2SAncestralSampler") и "s_churn" > 0 (от 0.0)
    }
    '''
    input_image_list = []
    for byte_image in init_img_binary_data_list:
        input_image_list.append(Image.open(io.BytesIO(byte_image)))
    binary_data_list = []
    result = web_api.img2img(images=input_image_list, prompt=caption, **params)
    img_byte_arr = io.BytesIO()
    result.image.save(img_byte_arr, format='PNG')
    binary_data = img_byte_arr.getvalue()
    return binary_data

def webui_img2img_controlnet(init_img_binary_data_list: List[bytes], init_img_binary_data_controlnet: bytes, caption: str, params: dict) -> List[bytes]:
    '''
    params={
        "host": "127.0.0.1", #хост где запущен webui
        "port": 7860, #порт где запущен webui
        "save": False, #сохранять картинки после генерации в папку save_dir
        "save_dir": "", #куда сохранять
        "negative_prompt": "", #антипромт
        "mask_image": None, #маска для inpainting
        "module": "None", #препроцессор
        "model": "None", #модель
        "weight": 1.0, #при низком получаются более размытые изображения (от 0.0 до 30.0)
        "resize_mode": str = "Resize and Fill", #(Resize and Fill, Crop and Resize, Just Resize)
        "control_mode": int = 0, #(0 - Баланс, 1 - Мой промт важнее, 2 - контролнет важнее)
        "pixel_perfect": bool = False, #делает изображение более четким
    }
    '''
    input_image_list = []
    for byte_image in init_img_binary_data_list:
        input_image_list.append(Image.open(io.BytesIO(byte_image)))
    image = Image.open(io.BytesIO(init_img_binary_data_controlnet))
    binary_data_list = []
    result = web_api.img2img_controlnet(prompt=caption, input_image=image, images=input_image_list, **params)
    for img in result.images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        binary_data_list.append(img_byte_arr.getvalue())
    return binary_data_list

def webui_upscaler(init_img_binary_data: bytes, params: dict) -> List[bytes]:
    '''
    params={
        "host": "127.0.0.1", #хост где запущен webui
        "port": 7860, #порт где запущен webui
        "save": False, #сохранять картинки после генерации в папку save_dir
        "save_dir": "", #куда сохранять
        "resize_mode": 0,
        "show_extras_results": True,
        "gfpgan_visibility": 0,
        "codeformer_visibility": 0,
        "codeformer_weight": 0,
        "upscaling_resize": 2, #множитель увеличения изображения
        "upscaling_resize_w": 512,
        "upscaling_resize_h": 512,
        "upscaling_crop": True,
        "upscaler_1": "ESRGAN_4x", # вид апскейлера ("Lanczos", "Nearest", "LDSR", "BSRGAN", "ESRGAN_4x", "R-ESRGAN General 4xV3", "ScuNET GAN", "ScuNET PSNR", "SwinIR 4x")
        "upscaler_2": "None",
        "extras_upscaler_2_visibility": 0,
        "upscale_first": False,
    }
    '''
    image = Image.open(io.BytesIO(init_img_binary_data))
    binary_data_list = []
    result = web_api.extra_single_image(image=image, **params)
    for img in result.images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        binary_data_list.append(img_byte_arr.getvalue())
    return binary_data_list