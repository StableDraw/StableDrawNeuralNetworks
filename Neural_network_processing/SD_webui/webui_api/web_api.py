import webuiapi

def text2img(
        host="127.0.0.1",
        port=7860, #хост и порт где запущен webui
        save=False,
        save_dir="",
        enable_hr=False, #использовать hr_upscaler после генерации
        denoising_strength=0.7, #(от 0.0 до 1.0)
        firstphase_width=0,
        firstphase_height=0,
        hr_scale=2,
        hr_upscaler=webuiapi.HiResUpscaler.Latent, #(Latent, LatentAntialiased, LatentBicubic, LatentBicubicAntialiased, LatentNearest, LatentNearestExact, Lanczos, Nearest, ESRGAN_4x, LDSR, ScuNET_GAN, ScuNET_PSNR, SwinIR_4x)
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        prompt="",
        seed=-1,
        subseed_strength=0.0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        sampler_name="Euler a", #(Euler a, Euler, DPM2, DPM2 a, DDIM, ...)
        batch_size=1,
        n_iter=1,
        steps=20,
        cfg_scale=7.0, #Размер cfg (от 0.0 до 30.0)
        width=512,
        height=512,
        restore_faces=False, #делает более реалистичные лица
        tiling=False, #делает изображение, все грани которого не содержат каких-то четких элементов.
        negative_prompt="",
        eta=1.0, #Только для обработчика "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
        s_churn=0, #Только для обработчиков "EulerEDMSampler" или "HeunEDMSampler" (от 0.0 до 1.0)
        s_tmax=0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы меньше этого значения (от 0.0 до "sigma_max" и < "s_tmax")
        s_tmin=0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы больше этого значения (от "sigma_min" до "sigma_max" и > "s_tmin")
        s_noise=1, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler" или "EulerAncestralSampler" или "DPMPP2SAncestralSampler") и "s_churn" > 0 (от 0.0)
):
    api = webuiapi.WebUIApi(host, port)
    params = {
        "enable_hr": enable_hr,
        "hr_scale": hr_scale,
        "hr_upscaler": hr_upscaler,
        "hr_second_pass_steps": hr_second_pass_steps,
        "hr_resize_x": hr_resize_x,
        "hr_resize_y": hr_resize_y,
        "denoising_strength": denoising_strength,
        "firstphase_width": firstphase_width,
        "firstphase_height": firstphase_height,
        "prompt": prompt,
        "seed": seed,
        "subseed_strength": subseed_strength,
        "seed_resize_from_h": seed_resize_from_h,
        "seed_resize_from_w": seed_resize_from_w,
        "batch_size": batch_size,
        "n_iter": n_iter,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "restore_faces": restore_faces,
        "tiling": tiling,
        "negative_prompt": negative_prompt,
        "eta": eta,
        "s_churn": s_churn,
        "s_tmax": s_tmax,
        "s_tmin": s_tmin,
        "s_noise": s_noise,
        "sampler_name": sampler_name,
    }
    result = api.txt2img(**params)
    if (save == True):
         for i in range(len(result.images)):
            result.images[i].save(f'{save_dir}{i}.png')
    return result

def text2img_controlnet(
        host="127.0.0.1",
        port=7860, #хост и порт где запущен webui
        save=False,
        save_dir="",
        prompt="",
        negative_prompt="",
        # Controlnet
        input_image = None,
        mask_image = None,
        module: str = "None", #препроцессор
        model: str = "None", #модель
        weight: float = 1.0,
        resize_mode: str = "Resize and Fill", #(Resize and Fill, Crop and Resize, Just Resize)
        control_mode: int = 0, #(0 - Баланс, 1 - Мой промт важнее, 2 - контролнет важнее)
        pixel_perfect: bool = False,
):
    api = webuiapi.WebUIApi(host, port)
    controlnet = webuiapi.ControlNetUnit(
        input_image = input_image,
        mask = mask_image,
        module = module,
        model = model,
        weight = weight,
        resize_mode = resize_mode,
        control_mode = control_mode,
        pixel_perfect = pixel_perfect,
    )
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "controlnet_units": [controlnet],
    }
    result = api.txt2img(**params)
    if (save == True):
         for i in range(len(result.images)):
            result.images[i].save(f'{save_dir}{i}.png')
    return result

def img2img(
        host="127.0.0.1",
        port=7860, #хост и порт где запущен webui
        save=False,
        save_dir="",
        images=[],  # list of PIL Image
        resize_mode=0,
        denoising_strength=0.75,
        image_cfg_scale=1.5,
        mask_image=None,  # PIL Image mask для inpainting
        mask_blur=4,
        inpainting_fill=0,
        inpaint_full_res=True,
        inpaint_full_res_padding=0,
        inpainting_mask_invert=0,
        initial_noise_multiplier=1,
        prompt="",
        seed=-1,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        sampler_name="Euler a", #(Euler a, Euler, DPM2, DPM2 a, DDIM, ...)
        batch_size=1,
        n_iter=1,
        steps=20,
        cfg_scale=7.0, #Размер cfg (от 0.0 до 30.0)
        width=512,
        height=512,
        restore_faces=False,
        tiling=False, #делает изображение, все грани которого не содержат каких-то четких элементов.
        negative_prompt="",
        eta=1.0,
        s_churn=0,
        s_tmax=0,
        s_tmin=0,
        s_noise=1,
):
    api = webuiapi.WebUIApi(host, port)
    params = {
        "images": images,
        "resize_mode": resize_mode,
        "denoising_strength": denoising_strength,
        "mask_blur": mask_blur,
        "inpainting_fill": inpainting_fill,
        "inpaint_full_res": inpaint_full_res,
        "inpaint_full_res_padding": inpaint_full_res_padding,
        "inpainting_mask_invert": inpainting_mask_invert,
        "initial_noise_multiplier": initial_noise_multiplier,
        "prompt": prompt,
        "seed": seed,
        "seed_resize_from_h": seed_resize_from_h,
        "seed_resize_from_w": seed_resize_from_w,
        "batch_size": batch_size,
        "n_iter": n_iter,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "image_cfg_scale": image_cfg_scale,
        "width": width,
        "height": height,
        "restore_faces": restore_faces,
        "tiling": tiling,
        "negative_prompt": negative_prompt,
        "eta": eta,
        "s_churn": s_churn,
        "s_tmax": s_tmax,
        "s_tmin": s_tmin,
        "s_noise": s_noise,
        "sampler_name": sampler_name,
    }
    if mask_image is not None:
            params["mask_image"] = mask_image
    result = api.img2img(**params)
    if (save == True):
         for i in range(len(result.images)):
            result.images[i].save(f'{save_dir}{i}.png')
    return result

def extra_single_image(
        image,  # PIL Image
        host="127.0.0.1",
        port=7860, #хост и порт где запущен webui
        save=False,
        save_dir="",
        resize_mode=0,
        show_extras_results=True,
        gfpgan_visibility=0,
        codeformer_visibility=0,
        codeformer_weight=0,
        upscaling_resize=2, #множитель увеличения изображения
        upscaling_resize_w=512,
        upscaling_resize_h=512,
        upscaling_crop=True,
        upscaler_1="ESRGAN_4x", # вид апскейлера ("Lanczos", "Nearest", "LDSR", "BSRGAN", "ESRGAN_4x", "R-ESRGAN General 4xV3", "ScuNET GAN", "ScuNET PSNR", "SwinIR 4x")
        upscaler_2="None",
        extras_upscaler_2_visibility=0,
        upscale_first=False,
):
    api = webuiapi.WebUIApi(host, port)
    params = {
            "resize_mode": resize_mode,
            "show_extras_results": show_extras_results,
            "gfpgan_visibility": gfpgan_visibility,
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
            "upscaling_resize": upscaling_resize,
            "upscaling_resize_w": upscaling_resize_w,
            "upscaling_resize_h": upscaling_resize_h,
            "upscaling_crop": upscaling_crop,
            "upscaler_1": upscaler_1,
            "upscaler_2": upscaler_2,
            "extras_upscaler_2_visibility": extras_upscaler_2_visibility,
            "upscale_first": upscale_first,
            "image": image,
        }
    result = api.extra_single_image(**params)
    if (save == True):
         for i in range(len(result.images)):
            result.images[i].save(f'{save_dir}{i}.png')
    return result

def img2img_controlnet(
        host="127.0.0.1",
        port=7860, #хост и порт где запущен webui
        save=False,
        save_dir="",
        images=[],  # list of PIL Image
        prompt="",
        negative_prompt="",
        # Controlnet
        input_image = None,
        mask_image = None,
        module: str = "None", #препроцессор
        model: str = "None", #модель
        weight: float = 1.0,
        resize_mode: str = "Resize and Fill", #(Resize and Fill, Crop and Resize, Just Resize)
        control_mode: int = 0, #(0 - Баланс, 1 - Мой промт важнее, 2 - контролнет важнее)
        pixel_perfect: bool = False,
):
    api = webuiapi.WebUIApi(host, port)
    controlnet = webuiapi.ControlNetUnit(
        input_image = input_image,
        mask = mask_image,
        module = module,
        model = model,
        weight = weight,
        resize_mode = resize_mode,
        control_mode = control_mode,
        pixel_perfect = pixel_perfect,
    )
    params = {
        "images": images,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "controlnet_units": [controlnet],
    }
    if mask_image is not None:
            params["mask_image"] = mask_image
    result = api.img2img(**params)
    if (save == True):
         for i in range(len(result.images)):
            result.images[i].save(f'{save_dir}{i}.png')
    return result