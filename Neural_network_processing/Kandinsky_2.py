import os
import torch
from huggingface_hub import hf_hub_url, cached_download
from copy import deepcopy
from omegaconf.dictconfig import DictConfig

from kandinsky2.configs import CONFIG_2_0, CONFIG_2_1
from kandinsky2.kandinsky2_model import Kandinsky2
from kandinsky2.kandinsky2_1_model import Kandinsky2_1
from kandinsky2.kandinsky2_2_model import Kandinsky2_2

ckpt_dir = "C:\\Stable-Draw\\Kandinsky-2\\checkpoints\\kandinsky2"

def get_kandinsky2_0(device, task_type = "text2img", use_auth_token = None):
    cache_dir = os.path.join(ckpt_dir, "2_0")
    config = deepcopy(CONFIG_2_0)
    if task_type == "inpainting":
        model_name = "Kandinsky-2-0-inpainting.pt"
        config_file_url = hf_hub_url(repo_id="sberbank-ai/Kandinsky_2.0", filename=model_name)
    elif task_type == "text2img" or task_type == "img2img":
        model_name = "Kandinsky-2-0.pt"
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = model_name)
    else:
        raise ValueError("Доступны только text2img, img2img и inpainting")
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = model_name, use_auth_token = use_auth_token)
    cache_dir_text_en1 = os.path.join(cache_dir, "text_encoder1")
    for name in ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = f"text_encoder1/{name}")
        cached_download(config_file_url, cache_dir = cache_dir_text_en1, force_filename = name, use_auth_token = use_auth_token)
    cache_dir_text_en2 = os.path.join(cache_dir, "text_encoder2")
    for name in ["config.json", "pytorch_model.bin", "spiece.model", "special_tokens_map.json", "tokenizer_config.json"]:
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = f"text_encoder2/{name}")
        cached_download(config_file_url, cache_dir = cache_dir_text_en2, force_filename = name, use_auth_token = use_auth_token)
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.0", filename = "vae.ckpt")
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = "vae.ckpt", use_auth_token = use_auth_token)
    config["text_enc_params1"]["model_path"] = cache_dir_text_en1
    config["text_enc_params2"]["model_path"] = cache_dir_text_en2
    config["tokenizer_name1"] = cache_dir_text_en1
    config["tokenizer_name2"] = cache_dir_text_en2
    config["image_enc_params"]["params"]["ckpt_path"] = os.path.join(cache_dir, "vae.ckpt")
    unet_path = os.path.join(cache_dir, model_name)
    model = Kandinsky2(config, unet_path, device, task_type)
    return model

def get_kandinsky2_1(device, task_type = "text2img", use_auth_token=None, use_flash_attention=False):
    cache_dir = os.path.join(ckpt_dir, "2_1")
    config = DictConfig(deepcopy(CONFIG_2_1))
    config["model_config"]["use_flash_attention"] = use_flash_attention
    if task_type == "text2img" or task_type == "img2img":
        model_name = "decoder_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = model_name)
    elif task_type == "inpainting":
        model_name = "inpainting_fp16.ckpt"
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = model_name)
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = model_name, use_auth_token = use_auth_token)
    prior_name = "prior_fp16.ckpt"
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = prior_name)
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = prior_name, use_auth_token = use_auth_token)
    cache_dir_text_en = os.path.join(cache_dir, "text_encoder")
    for name in ["config.json", "pytorch_model.bin", "sentencepiece.bpe.model", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = f"text_encoder/{name}")
        cached_download(config_file_url, cache_dir = cache_dir_text_en, force_filename = name, use_auth_token = use_auth_token)
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = "movq_final.ckpt")
    cached_download(config_file_url, cache_dir=cache_dir, force_filename = "movq_final.ckpt", use_auth_token = use_auth_token)
    config_file_url = hf_hub_url(repo_id = "sberbank-ai/Kandinsky_2.1", filename = "ViT-L-14_stats.th")
    cached_download(config_file_url, cache_dir = cache_dir, force_filename = "ViT-L-14_stats.th", use_auth_token = use_auth_token)
    config["tokenizer_name"] = cache_dir_text_en
    config["text_enc_params"]["model_path"] = cache_dir_text_en
    config["prior"]["clip_mean_std_path"] = os.path.join(cache_dir, "ViT-L-14_stats.th")
    config["image_enc_params"]["ckpt_path"] = os.path.join(cache_dir, "movq_final.ckpt")
    cache_model_name = os.path.join(cache_dir, model_name)
    cache_prior_name = os.path.join(cache_dir, prior_name)
    model = Kandinsky2_1(config, cache_model_name, cache_prior_name, device, task_type=task_type)
    return model

def get_kandinsky2(device, task_type = "text2img", use_auth_token = None, model_version = "2.2", use_flash_attention = False):
    if model_version == "2.0":
        model = get_kandinsky2_0(device, task_type = task_type, use_auth_token = use_auth_token)
    elif model_version == "2.1":
        model = get_kandinsky2_1(device, task_type = task_type, use_auth_token = use_auth_token, use_flash_attention = use_flash_attention)
    elif model_version == "2.2":
        model = Kandinsky2_2(device = device, task_type = task_type)
    else:
        raise ValueError("Доступны только 2.0, 2.1 и 2.2")
    return model

def Kandinsky2_text_to_image(prompt, opt):
    if opt["low_vram_mode"] == True:
        device = "cpu"
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if opt["version"] == "Kandinsky2.0":
        model = get_kandinsky2(device, task_type = "text2img", model_version = "2.0", use_flash_attention = opt["use_flash_attention"])
        binary_data_list = model.generate_text2img(prompt, batch_size = opt["num_cols"], h = opt["h"], w = opt["w"], num_steps = opt["steps"], denoised_type = opt["denoised_type"], dynamic_threshold_v = opt["dynamic_threshold_v"], sampler = opt["sampler"], ddim_eta = opt["ddim_eta"], guidance_scale = opt["guidance_scale"], seed = opt["seed"])
    elif opt["version"] == "Kandinsky2.1":
        model = get_kandinsky2(device = device, task_type = "text2img", model_version = "2.1", use_flash_attention = opt["use_flash_attention"])
        binary_data_list = model.generate_text2img(prompt, num_steps = opt["steps"], batch_size = opt["num_cols"], guidance_scale = opt["guidance_scale"], h = opt["h"], w = opt["w"], sampler = opt["sampler"], ddim_eta = opt["ddim_eta"], prior_cf_scale = opt["prior_scale"], prior_steps = str(opt["prior_steps"]), negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"], progress = opt["progress"])
    elif opt["version"] == "Kandinsky2.2":
        if opt["ControlNET"] == True:
            model = get_kandinsky2(device, task_type = "text2imgCN", model_version = "2.2")
            binary_data_list = model.generate_text2imgCN(prompt, decoder_steps = opt["steps"], batch_size = opt["num_cols"], prior_steps = opt["prior_steps"], prior_guidance_scale = opt["prior_scale"], decoder_guidance_scale = opt["guidance_scale"], h = opt["h"], w = opt["w"], negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"])
        else:
            model = get_kandinsky2(device, task_type = "text2img", model_version = "2.2")
            binary_data_list = model.generate_text2img(prompt, decoder_steps = opt["steps"], batch_size = opt["num_cols"], prior_steps = opt["prior_steps"], prior_guidance_scale = opt["prior_scale"], decoder_guidance_scale = opt["guidance_scale"], h = opt["h"], w = opt["w"], negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"])
    else:
        raise ValueError("Доступны только версии Kandinsky2.0, Kandinsky2.1 и Kandinsky2.2")
    return binary_data_list

def Kandinsky2_image_to_image(binary_data, prompt, opt):
    if opt["low_vram_mode"] == True:
        device = "cpu"
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if opt["version"] == "Kandinsky2.0":
        model = get_kandinsky2(device, task_type = "img2img", model_version = "2.0")
        binary_data_list = model.generate_img2img(prompt = prompt, binary_data = binary_data, batch_size = opt["num_cols"], strength = opt["i2i_strength"], num_steps = opt["steps"], guidance_scale = opt["guidance_scale"], progress = opt["progress"], dynamic_threshold_v = opt["dynamic_threshold_v"], h = opt["h"], w = opt["w"], denoised_type = opt["denoised_type"], sampler = opt["sampler"], ddim_eta = opt["ddim_eta"], seed = opt["seed"], custom_orig_size = opt["custom_orig_size"])
    elif opt["version"] == "Kandinsky2.1":
        model = get_kandinsky2(device, task_type = "img2img", model_version = "2.1")
        binary_data_list = model.generate_img2img(prompt = prompt, binary_data = binary_data, strength = opt["i2i_strength"], num_steps = opt["steps"], batch_size = opt["num_cols"], guidance_scale = opt["guidance_scale"], h = opt["h"], w = opt["w"], sampler = opt["sampler"], ddim_eta = opt["ddim_eta"], prior_cf_scale = opt["prior_scale"], prior_steps = str(opt["prior_steps"]), negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"], custom_orig_size = opt["custom_orig_size"], progress = opt["progress"])
    elif opt["version"] == "Kandinsky2.2":
        if opt["ControlNET"] == False:
            if opt["Depth"] == False:
                model = get_kandinsky2(device, task_type = "img2img", model_version = "2.2")
                binary_data_list = model.generate_img2img(prompt = prompt, binary_data = binary_data, strength = opt["i2i_strength"], batch_size = opt["num_cols"], decoder_steps = opt["steps"], prior_steps = opt["prior_steps"], decoder_guidance_scale = opt["guidance_scale"], prior_guidance_scale = opt["prior_scale"], h = opt["h"], w = opt["w"], negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"], custom_orig_size = opt["custom_orig_size"])
            else:
                model = get_kandinsky2(device, task_type = "depth2img", model_version = "2.2")
                binary_data_list = model.generate_depth2img(prompt, binary_data = binary_data, decoder_steps = opt["steps"], batch_size = opt["num_cols"], prior_steps = opt["prior_steps"], prior_guidance_scale = opt["prior_scale"], decoder_guidance_scale = opt["guidance_scale"], h = opt["h"], w = opt["w"], negative_prior_prompt = opt["negative_prior_prompt"], seed = opt["seed"], custom_orig_size = opt["custom_orig_size"])
        else:
            model = get_kandinsky2(device, task_type = "img2imgCN", model_version = "2.2")
            binary_data_list = model.generate_img2imgCN(prompt = prompt, binary_data = binary_data, strength = opt["i2i_strength"], batch_size = opt["num_cols"], decoder_steps = opt["steps"], prior_steps = opt["prior_steps"], decoder_guidance_scale = opt["guidance_scale"], prior_guidance_scale = opt["prior_scale"], h = opt["h"], w = opt["w"], negative_prior_prompt = opt["negative_prior_prompt"], seed = opt["seed"], custom_orig_size = opt["custom_orig_size"], prior_strength = opt["i2i_prior_strength"], negative_prior_strength = opt["i2i_negative_prior_strength"])
    else:
        raise ValueError("Доступны только версии Kandinsky2.0, Kandinsky2.1 и Kandinsky2.2")
    return binary_data_list

def Kandinsky2_inpainting(binary_data, mask_binary_data, prompt, opt):
    if opt["low_vram_mode"] == True:
        device = "cpu"
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if opt["version"] == "Kandinsky2.0":
        model = get_kandinsky2(device, task_type = "inpainting", model_version = "2.0")
        binary_data_list = model.generate_inpainting(prompt, binary_data, mask_binary_data, batch_size = opt["num_cols"], num_steps = opt["steps"], guidance_scale = opt["guidance_scale"], progress = opt["progress"], dynamic_threshold_v = opt["dynamic_threshold_v"], denoised_type = opt["denoised_type"], sampler = opt["sampler"], ddim_eta = opt["ddim_eta"], seed = opt["seed"])
    elif opt["version"] == "Kandinsky2.1":
        model = get_kandinsky2(device, task_type = "inpainting", model_version = "2.1", use_flash_attention = False)
        binary_data_list = model.generate_inpainting(prompt, binary_data, mask_binary_data, num_steps = opt["steps"], batch_size = opt["num_cols"], guidance_scale = opt["guidance_scale"], sampler = opt["sampler"], ddim_eta = opt["ddim_eta"], prior_cf_scale = opt["prior_scale"], prior_steps = str(opt["prior_steps"]), negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"], progress = opt["progress"])
    elif opt["version"] == "Kandinsky2.2":
        model = get_kandinsky2(device, task_type = "inpainting", model_version = "2.2")
        binary_data_list = model.generate_inpainting(prompt, binary_data, mask_binary_data, batch_size = opt["num_cols"], decoder_steps = opt["steps"], prior_steps = opt["prior_steps"], decoder_guidance_scale = opt["guidance_scale"], prior_guidance_scale = opt["prior_scale"], negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"])
    else:
        raise ValueError("Доступны только версии Kandinsky2.0, Kandinsky2.1 и Kandinsky2.2")
    return binary_data_list

def Kandinsky2_stylization(content_binary_data, style_binary_data, prompt, opt):
    if opt["low_vram_mode"] == True:
        device = "cpu"
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_kandinsky2(device, task_type = "text2img", model_version = "2.2")
    binary_data_list = model.generate_stylization(prompt, content_binary_data, style_binary_data, batch_size = opt["num_cols"], decoder_steps = opt["steps"], prior_steps = opt["prior_steps"], decoder_guidance_scale = opt["guidance_scale"], prior_guidance_scale = opt["prior_scale"], negative_prior_prompt = opt["negative_prior_prompt"], prompt_weight = opt["prompt_weight"], style_size_as_content = opt["style_size_as_content"], content_weight = opt["content_weight"], style_weight = opt["style_weight"], seed = opt["seed"])
    return binary_data_list

def Kandinsky2_mix_images(image1_binary_data, image2_binary_data, prompt_image1, prompt_image2, opt):
    if opt["low_vram_mode"] == True:
        device = "cpu"
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if opt["version"] == "Kandinsky2.1":
        model = get_kandinsky2(device, task_type = "text2img", model_version = "2.1", use_flash_attention = False)
        binary_data_list = model.mix_images(image1_binary_data, image2_binary_data, prompt_image1, prompt_image2, weights = [opt["first_prompt_weight"], opt["first_image_weight"], opt["second_image_weight"], opt["second_prompt_weight"]], num_steps = opt["steps"], batch_size = opt["num_cols"], guidance_scale = opt["guidance_scale"], h = opt["h"], w = opt["w"], sampler = opt["sampler"], ddim_eta = opt["ddim_eta"], prior_cf_scale = opt["prior_scale"], prior_steps = str(opt["prior_steps"]), negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"], custom_orig_size = opt["custom_orig_size"], progress = opt["progress"])
    elif opt["version"] == "Kandinsky2.2":
        model = get_kandinsky2(device, task_type = "text2img", model_version = "2.2")
        binary_data_list = model.mix_images(image1_binary_data, image2_binary_data, prompt_image1, prompt_image2, weights = [opt["first_prompt_weight"], opt["first_image_weight"], opt["second_image_weight"], opt["second_prompt_weight"]], decoder_steps = opt["steps"], prior_steps = opt["prior_steps"], batch_size = opt["num_cols"], decoder_guidance_scale = opt["guidance_scale"], prior_guidance_scale = opt["prior_scale"], h = opt["h"], w = opt["w"], negative_prior_prompt = opt["negative_prior_prompt"], negative_decoder_prompt = opt["negative_prompt"], seed = opt["seed"], custom_orig_size = opt["custom_orig_size"])
    else:
        raise ValueError("Доступны только версии Kandinsky2.1 и Kandinsky2.2")
    return binary_data_list

if __name__ == "__main__":
    '''
    params = {
        "version": "Kandinsky2.2", #("Kandinsky2.0", "Kandinsky2.1", "Kandinsky2.2")
        "ControlNET": False, #Только для "Kandinsky2.2"
        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
        "use_flash_attention": False, #Только для "Kandinsky"
        "steps": 50,
        "prior_steps": 25, #Только для Kandinsky > 2.0
        "num_cols": 1,
        "guidance_scale": 7,
        "prior_scale": 4, #Только для Kandinsky > 2.0
        "progress": True, #Только для Kandinsky < 2.2 и обработчика "p_sampler"
        "dynamic_threshold_v": 99.5, #Только для "Kandinsky2.0" и "dynamic_threshold"
        "denoised_type": "dynamic_threshold", #("dynamic_threshold", "clip_denoised") только для "Kandinsky2.0"
        "h": 1024, #512 для Kandinsky 2.0, 768 для Kandinsky 2.1 и 1024 для Kandinsky 2.2
        "w": 1024, #512 для Kandinsky 2.0, 768 для Kandinsky 2.1 и 1024 для Kandinsky 2.2
        "sampler": "ddim_sampler", #("ddim_sampler", "plms_sampler", "p_sampler") Только для Kandinsky < 2.2
        "ddim_eta": 0.05, #только для обработчика "ddim_sampler" и Kandinsky < 2.2
        "seed": 42, #Инициализирующее значение (может быть от 0 до 1000000000)
        "negative_prior_prompt": "", #Только для Kandinsky > 2.0
        "negative_prompt": "", #Только для Kandinsky > 2.0
    }
    #prompt = "A teddy bear на красной площади"
    prompt = "A robot, 4k photo"
    params["negative_prior_prompt"] = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
    images = Kandinsky2_text_to_image(prompt, params)
    iii = 0
    for image in images:
        iii += 1
        with open("output\\2.2\\t2i\\i_" + str(iii) + ".png", "wb") as f:
            f.write(image)
    '''
    '''
    params = {
        "version": "Kandinsky2.2", #("Kandinsky2.0", "Kandinsky2.1", "Kandinsky2.2")
        "ControlNET": False, #Только для "Kandinsky2.2"
        "Depth": True, #Использовать дополнительный слой глубины (только для "Kandinsky2.2" ControlNET)
        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
        "i2i_prior_strength": 0.85, #Только для "Kandinsky2.2" с "ControlNET"
        "i2i_negative_prior_strength": 1.0, #Только для "Kandinsky2.2" с "ControlNET"
        "i2i_strength": 0.4,
        "steps": 100,
        "prior_steps": 25, #Только для Kandinsky > 2.0
        "num_cols": 1,
        "custom_orig_size": True, #Если применён, то меняет размеры входного изображения на "w" и "h", иначе оставляет равними размерам желаемого изображения
        "w": 768, #Ширина входного изображения, если установлен параметр "custom_orig_width" (от 16) Только для Kandinsky > 2.0
        "h": 768, #Высота входного изображения, если установлен параметр "custom_orig_width" (от 16) Только для Kandinsky > 2.0
        "progress": True, #Только для Kandinsky < 2.2 и обработчика "p_sampler"
        "guidance_scale": 4,
        "prior_scale": 4, #Только для Kandinsky > 2.0
        "dynamic_threshold_v": 99.5, #Только для "Kandinsky2.0" и "dynamic_threshold"
        "denoised_type": "dynamic_threshold", #("dynamic_threshold", "clip_denoised") только для "Kandinsky2.0"
        "sampler": "ddim_sampler", #("ddim_sampler", "plms_sampler", "p_sampler") Только для Kandinsky < 2.2
        "ddim_eta": 0.05, #только для обработчика "ddim_sampler" и Kandinsky < 2.2
        "seed": 42,
        "negative_prior_prompt": "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature", #Только для Kandinsky > 2.0
        "negative_prompt": "" #Только для Kandinsky > 2.0
    }
    prompt = "A robot, 4k photo"
    with open("C:\\Stable-Draw\\Kandinsky-2\\cat.png", "rb") as f:
        init_img_binary_data = f.read()
    image = Kandinsky2_image_to_image(init_img_binary_data, prompt, params)[0]
    with open("output\\2.2\\i2i\\i.png", "wb") as f:
        f.write(image)
    '''
    '''
    params = {
        "version": "Kandinsky2.2",    
        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
        "num_cols": 1, 
        "steps": 50,
        "prior_steps": 25, #Только для Kandinsky > 2.0
        "guidance_scale": 4,
        "prior_scale": 4, #Только для Kandinsky > 2.0
        "progress": True, #Только для Kandinsky < 2.2 и обработчика "p_sampler"
        "dynamic_threshold_v": 99.5, #Только для "Kandinsky2.0" и "dynamic_threshold"
        "denoised_type": "dynamic_threshold", #("dynamic_threshold", "clip_denoised") только для "Kandinsky2.0"
        "sampler": "ddim_sampler", #("ddim_sampler", "plms_sampler", "p_sampler") Только для Kandinsky < 2.2
        "ddim_eta": 0.05, #только для обработчика "ddim_sampler" и Kandinsky < 2.2
        "seed": 42,
        "negative_prior_prompt": "", #Только для Kandinsky > 2.0
        "negative_prompt": "" #Только для Kandinsky > 2.0
    }
    prompt = "A cat in a hat"
    with open("C:\\Stable-Draw\\Kandinsky-2\\cat.png", "rb") as f:
        init_img_binary_data = f.read()
    with open("C:\\Stable-Draw\\Kandinsky-2\\mask.png", "rb") as f:
        mask_binary_data = f.read()
    image = Kandinsky2_inpainting(init_img_binary_data, mask_binary_data, prompt, params)[0]
    with open("output\\2.2\\inpainting\\3i.png", "wb") as f:
        f.write(image)
    '''
    '''
    params = {
        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
        "num_cols": 1, 
        "steps": 50,
        "prior_steps": 25,
        "guidance_scale": 4,
        "prior_scale": 4,
        "style_size_as_content": True, #Изменять размер изображения стиля под размеры изображения контента
        "seed": 42,
        "prompt_weight": 0.3, #Вес описания (в сумме, как я понял, все 3 веса должны равняться 1)
        "content_weight": 0.3, #Вес контента (в сумме, как я понял, все 3 веса должны равняться 1)
        "style_weight": 0.4, #Вес стиля (в сумме, как я понял, все 3 веса должны равняться 1)
        "negative_prior_prompt": ""
    }
    prompt = "a cat"
    with open("C:\\Stable-Draw\\Kandinsky-2\\cat.png", "rb") as f:
        content_binary_data = f.read()
    with open("C:\\Stable-Draw\\Kandinsky-2\\starry_night.jpeg", "rb") as f:
        style_binary_data = f.read()
    image = Kandinsky2_stylization(content_img_binary_data, style_binary_data, prompt, params)[0]
    with open("output\\2.2\\stylization\\5i.png", "wb") as f:
        f.write(image)
    '''
    params = {
        "version": "Kandinsky2.2", #("Kandinsky2.1", "Kandinsky2.2")
        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
        "steps": 100,
        "prior_steps": 25,
        "first_image_weight": 0.25, #Вес первого изображения (в сумме, как я понял, все 3 веса должны равняться 1)
        "second_image_weight": 0.25, #Вес второго изображения (в сумме, как я понял, все 3 веса должны равняться 1)
        "first_prompt_weight": 0.25, #Вес первого описания (в сумме, как я понял, все 3 веса должны равняться 1)
        "second_prompt_weight": 0.25, #Вес второго описания (в сумме, как я понял, все 3 веса должны равняться 1)
        "num_cols": 1,
        "custom_orig_size": True, #Если применён, то меняет размеры входного изображения на "w" и "h", иначе оставляет равными w = max(w1, w1), h = max(h1, h2) от обоих изображений
        "w": 1024, #Ширина входного изображения, если установлен параметр "custom_orig_width" (от 16)
        "h": 1024, #Высота входного изображения, если установлен параметр "custom_orig_width" (от 16)
        "progress": True, #Только для Kandinsky 2.1 и обработчика "p_sampler"
        "guidance_scale": 4,
        "prior_scale": 4,
        "sampler": "ddim_sampler", #("ddim_sampler", "plms_sampler", "p_sampler") Только для Kandinsky 2.1
        "ddim_eta": 0.05, #только для обработчика "ddim_sampler" и Kandinsky 2.1
        "seed": 42,
        "negative_prior_prompt": "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature", #Только для Kandinsky > 2.0
        "negative_prompt": ""
    }
    prompt1 = "red cat"
    prompt2 = "a man"
    with open("C:\\Stable-Draw\\Kandinsky-2\\cat.png", "rb") as f:
        img1_binary_data = f.read()
    with open("C:\\Stable-Draw\\Kandinsky-2\\man.png", "rb") as f:
        img2_binary_data = f.read()
    image = Kandinsky2_mix_images(img1_binary_data, img2_binary_data, prompt1, prompt2, params)[0]
    with open("output\\2.2\\mix\\i.png", "wb") as f:
        f.write(image)