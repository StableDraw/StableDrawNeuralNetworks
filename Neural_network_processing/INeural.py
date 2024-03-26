from typing import Optional, List

from Image_classifier import Get_image_class
from Translate import Translator
from Image_сolorization import Image_сolorizer
from Image_caption_generator import Gen_caption
from Delete_background import U2NET_Delete_background
from Delete_background import DIS_Delete_background



#from upscaler.MultichannelRealESRGAN.RealESRGAN import RealESRGAN_upscaler



#from upscaler.StableDiffusionx4Upscaler.StableDiffusionUpscaler import Stable_diffusion_upscaler
#from upscaler.StableDiffusionx2LatentUpscaler.StableDiffusionx2LatentUpscaler import Stable_diffusion_upscaler_xX
#from RealESRGAN import RealESRGAN_upscaler
from Stable_diffusion import Stable_diffusion_upscaler
from Stable_diffusion import Stable_diffusion_upscaler_xX

from Stable_diffusionXL import Stable_diffusion_XL_image_to_image
from Kandinsky_2 import Kandinsky2_text_to_image
from Stable_diffusion import Stable_diffusion_text_to_image as Stable_diffusion_2_0_text_to_image
from Stable_diffusionXL import Stable_diffusion_XL_text_to_image
from Stable_diffusion import Stable_diffusion_image_to_image as Stable_diffusion_2_0_image_to_image
from Stable_diffusion import Stable_diffusion_depth_to_image as Stable_diffusion_2_0_depth_to_image
from Kandinsky_2 import Kandinsky2_image_to_image
from Stable_diffusion import Stable_diffusion_inpainting
from Kandinsky_2 import Kandinsky2_inpainting
from Kandinsky_2 import Kandinsky2_stylization
from Kandinsky_2 import Kandinsky2_mix_images
from webui import webui_upscaler, webui_img2img, webui_text2img


#Колоризатор (4 модели с одинаковыми параметрами) (будет заменён в скором будущем)
#Принимает изображение и параметры. Возвращает изображение
def colorizer(init_img_binary_data: bytes, params: dict) -> bytes:
    '''
    params = {
        "ckpt": "ColorizeArtistic_gen",    #Выбор модели ("ColorizeArtistic_gen", "ColorizeArtistic_gen_GrayScale", "ColorizeArtistic_gen_Sketch", "ColorizeArtistic_gen_Sketch2Gray")
        "steps": 1,                        #Количество шагов обработки (минимум 1)
        "compare": False,                  #Сравнивать с оригиналом
        "artistic": True,                  #Дополнительная модель для обработки
        "render_factor": 12,               #Фактор обработки (от 7 до 45) (лучше 12)
        "post_process": True,              #Постобработка
        "clr_saturation_factor": 5,        #Коэффициент увеличения цветовой насыщенности (1 - не добавлять насыщенность)
        "line_color_limit": 100,           #минимальная яркость пикселя, при которой цветовая насыщенность увеличиваться не будет (меньше для цифровых рисунков, больше для рисунков карандашом. 1 если лайн абсолютно чёрный)
        "clr_saturate_every_step": True    #Повышать цветовую насыщенность после каждого шага (играет роль только если количество шагов обработки больше 1)
        #max_dim не учитывается, поскольку в колоризатор встроен внутренний даунсемплер
    }
    '''
    binary_data = Image_сolorizer(init_img_binary_data, params)
    return binary_data

#Удаление фона (2 модели с разными параметрами в зависимости от модели, читайте комментарии)
#Принимает изображение и параметры. Возвращает изображение
def delete_background(init_img_binary_data: bytes, params: dict) -> bytes:
    '''
    params = {
        "model": "DIS", #Доступно "U2NET" или "DIS"
        "RescaleT": 320, #Только для модели U2NET
        #Только для модели "DIS":
        "ckpt": "isnet.pth",        # Выбор впретренированных весов модели ("isnet.pth", "isnet-general-use.pth")
        "interm_sup": False,        # Указать, активировать ли контроль промежуточных функций
        "model_digit": True,        # Выберите точность с плавающей запятой (устанавливает False или True точность числа с плавающей запятой)
        "seed": 0,                  # Инициализирующее значение
        "cache_size": [1024, 1024], # Кешированное входное пространственное разрешение, можно настроить на другой размер
        "input_size": [1024, 1024], # Входной пространственный размер модели, обычно используют одно и то же значение params["cache_size"], что означает, что мы больше не изменяем размер изображений
        "crop_size": [1024, 1024]   # Размер случайно обрезки из ввода, обычно он меньше, чем params["cache_size"], например, [920, 920] для увеличения данных
    #max_dim, возможно, даунсемплится по "*_size" параметрам выше. Не исследованно, и зависит от модели. По умолчанию, ограничений нет
    }
    '''
    if params["model"] == "U2NET":
        binary_data = U2NET_Delete_background(init_img_binary_data, params)
    else:
        binary_data = DIS_Delete_background(init_img_binary_data, params)
    return binary_data

#Апскейлер (3 группы моделей с разными параметрами, в зависимости от модели или группы моделей, внимательно читайте комментарии)
#Принимает изображение, описание (для RealESR моделей описание не нужно, но нужно для SD, то есть оставшихся) и параметры. Возвращает изображение
def upscaler(init_img_binary_data: bytes, caption: Optional[str], params: dict) -> bytes:
    '''
    #Апскейлер (3 разных апскейлера с разными параметрами в зависимости от модели, читайте комментарии)
    params = {
        "model": "StableDiffusionx4Upscaler",   #("StableDiffusionx4Upscaler", "StableDiffusionxLatentx2Upscaler", "RealESRGAN_x4plus" - модель x4 RRDBNet, "RealESRNet_x4plus" - модель x4 RRDBNet, "RealESRGAN_x4plus_anime_6B" - модель x4 RRDBNet с 6 блоками, "RealESRGAN_x2plus" - модель x2 RRDBNet, "realesr-animevideov3" - модель x4 VGG-стиля (размера XS), "realesr-general-x4v3" - модель x4 VGG-стиля (размера S))
        "steps": 50,                            #Шаги DDIM, от 2 до 250
        "ddim_eta": 0.0,                        #значения от 0.0 до 1.0, η = 0.0 соответствует детерминированной выборке
        "guidance_scale": 9.0,                  #от 0.1 до 30.0
        "ckpt": "x4-upscaler-ema.safetensors",  #выбор весов модели ("x4-upscaler-ema.safetensors", только для модели "StableDiffusionx4Upscaler")
        "sampler": "ddim_sampler",              #выбор обработчика ("ddim_sampler", "plms_sampler", "p_sampler", только для модели "StableDiffusionx4Upscaler")
        "seed": 42,                             #от 0 до 1000000
        "outscale": 4,                          #Величина того, во сколько раз увеличть разшрешение изображения (рекоммендуется 2 для моделей ("StableDiffusionxLatentx2Upscaler" и "RealESRGAN_x2plus") и 4 для остальных)
        "noise_augmentation": 20,               #от 0 до 350
        "negative_prompt": "",                  #отрицательное описание (если без него, то "")
        "verbose": False,                       #Не знаю что это
        "max_dim": pow(1024, 2),                #Максимальное разрешение ((для всех моделей, кроме "RealESRGAN_x2plus") и "outscale": 4), и pow(2048, 2) (для модели "RealESRGAN_x2plus" и "outscale": 2)
        #Только для моделей RealESR:
        "denoise_strength": 0.5,                #Сила удаления шума. 0 для слабого удаления шума (шум сохраняется), 1 для сильного удаления шума. Используется только для модели "realesr-general-x4v3"
        "tile": 0,                              #Размер плитки, 0 для отсутствия плитки во время тестирования, влияет на количество требуемой видеопамяти и скорость обработки
        "tile_pad": 10,                         #Заполнение плитки
        "pre_pad": 0,                           #Предварительный размер заполнения на каждой границе
        "face_enhance": False,                  #Использовать GFPGAN улучшения лиц
        "version": "RestoreFormer_GFPGAN",      #Версия модели для улучшения лиц. Только для моделей семейства RealESRGAN, если выбран "face_enhance: True. Возможне значения: "1.1", "1.2", "1.3", "1.4", "RestoreFormerGFPGAN", "RestoreFormer". Модель 1.1 тестовая, но способна колоризировать. Модель 1.2 обучена на большем количестве данных с предобработкой, не способна колоризировать, генерирует достаточно чёткие изображения с красивым магияжем, однако иногда результат генерации выглядит не натурально. Модель 1.3 основана на модели 1.2, генерирует более натурально выглядящие изображения, однако не такие чёткие, выдаёт лучие результаты на более низкокачественных изображениях, работает с относительно высококачественными изображениями, может иметь повторяющееся (дважды) восстановление. Модель 1.4 обеспечивает немного больше деталей и лучшую идентичность. Модель RestoreFormer создана специально для улучшения лиц, "RestoreFormer_GFPGAN" обеспечивает более чёткую, однако менее натуралистичную обработку и иногда создаёт артифакты.
        "input_is_latent": True,                #Скрытый ли вход. Только для моделей семейства RealESRGAN, если выбран "face_enhance: True и "version" от 1.1 до 1.4. Если выбран, то результат менее насыщенный и чёткий, но более наруральный
        "fp32": True,                           #Использовать точность fp32 во время вывода. По умолчанию fp16 (половинная точность)
        "alpha_upsampler": "realesrgan",        #Апсемплер для альфа-каналов. Варианты: realesrgan | bicubic
        "gpu-id": 0                             #Устройство gpu для использования (по умолчанию = 0) может быть 0, 1, 2 для обработки на нескольких GPU
    }
    '''
    if params["model"] == "StableDiffusionx4Upscaler":
        binary_data = Stable_diffusion_upscaler(init_img_binary_data, caption, params)
    elif params["model"] == "StableDiffusionxLatentx2Upscaler":
        binary_data = Stable_diffusion_upscaler_xX(init_img_binary_data, caption, params)
        '''
    elif "REALESR" in params["model"].upper():
        binary_data = RealESRGAN_upscaler(init_img_binary_data, params) #передаю путь к рабочей папке
        '''
    elif params["model"] == "webui_upscaler":
        binary_data = webui_upscaler(init_img_binary_data, params)
    else:
        raise ValueError("Доступны только \"StableDiffusionx4Upscaler\" и \"StableDiffusionx4Upscaler\"")
    return binary_data

#Генерация по тексту и изображению (Image to image) (4 разные нейронки, у них разные параметры, в комментариях всё указано, требуется максимум внимания)
#Принимает изображение, описание и параметры. Возвращает список изображений
def image_to_image(init_img_binary_data: bytes, caption: str, params: dict) -> List[bytes]:
    '''
    params = {
        #Параметры не для пользователя:
        "add_watermark": False, #Добавлять невидимую вотермарку
        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
        "max_dim": pow(2048, 2), # я не могу генерировать на своей видюхе картинки больше 2048 на 2048
        #Параметры для пользователя:
        "version": "SDXL-base-1.0", #Выбор версии: "SDXL-base-1.0", "SDXL-base-0.9" (недоступна для коммерческого использования), "SD-2.0", "SD-2.1", "SD-2.1-768", "SDXL-refiner-0.9" (недоступна для коммерческого использования, используется как модель 2 стадии, для первой непригодна),  "SDXL-refiner-1.0" (используется как модель 2 стадии, для первой непригодна), "Kandinsky2.0", "Kandinsky2.1", "Kandinsky2.2"
        "ControlNET": False, #Только для "Kandinsky2.2"
        "progress": True, #Только для Kandinsky < 2.2 и обработчика "p_sampler"
        "Depth": True, #Использовать дополнительный слой глубины (только для версий "Kandinsky2.2" ControlNET и версий "SD-2.0")
        "use_custom_ckpt": False, #Использовать свои веса для выбранной версии модели (для всех версий кроме SD-2.0)
        "custom_ckpt_name": "512-depth-ema.safetensors", #Имя кастомной модели, если выбран "use_custom_ckpt". Является обязательным параметром для версии SD-2.0. (SD-2.0: "512-depth-ema.safetensors" для Depth == True, и "sd-v1-1.safetensors", "sd-v1-1-full-ema.safetensors", "sd-v1-2.safetensors", "sd-v1-2-full-ema.safetensors", "sd-v1-3.safetensors", "sd-v1-3-full-ema.safetensors", "sd-v1-4.safetensors", "sd-v1-4-full-ema.safetensors", "sd-v1-5.safetensors", "sd-v1-5-full-ema.safetensors" для Depth == False)        
        "version2SDXL-refiner": False, #Только для версий SDXL-base: загрузить SDXL-refiner как модель для второй стадии обработки. Требует более длительной обработки и больше видеопамяти
        "seed": 42, #Инициализирующее значение (может быть от 0 до 1000000000)
        "negative_prompt": "", #Для всех моделей, кроме SDXL-base и Kandinsky 2.0: негативное описание
        "negative_prior_prompt": "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature", #Только для Kandinsky > 2.0
        "refiner": "SDXL-refiner-1.0", #Если "version2SDXL-refiner" выбран, то какую версию модели для второй стадии обработки загрузить: "SDXL-refiner-1.0", "SDXL-refiner-0.9"  (недоступна для коммерческого использования)
        "refinement_strength": 0.15, #Сила вклада обработки на второй стадии (от 0.0 до 1.0)
        "finish_denoising": True, #Завершить удаление шума рафинёром (только для моделей SDXL-base, если включён version2SDXL-refiner)
        "h": 1024, #Высота желаемого изображения (от 64 до 2048, должна быть кратна 64, для "SD-2.0" рекомендуется 512)
        "w": 1024, #Ширина желаемого изображения (от 64 до 2048, должна быть кратна 64, для "SD-2.0" рекомендуется 512)
        "f": 8, #Коэффициент понижающей дискретизации, чаще всего 8 или 16 (можно 4, тогда есть риск учетверения, но красиво и жрёт больше видеопамяти) (От 4 до 64), если (4 можно, если w * h <= 1024 * 1024, иначе > 4)
        "use_recommended_res": True, #Использовать рекомендованное для каждой модели разрешение генерации, вместо указанных выше
        "sampler": "EulerEDMSampler", #Обработчик (("EulerEDMSampler", "HeunEDMSampler", "EulerAncestralSampler", "DPMPP2SAncestralSampler", "DPMPP2MSampler", "LinearMultistepSampler") только для моделей кроме Kandinsky и SD-2.0), (("ddim_sampler", "plms_sampler", "p_sampler") Только для Kandinsky < 2.2 и моделей SD-2.0)
        "ddim_eta": 0.0, #Только для моделей ("SD-2.0" и Kandinsky < 2.2) и обработчика ddim, η (η = 0.0 соответствует детерминированной выборке)
        "dynamic_threshold_v": 99.5, #Только для "Kandinsky2.0" и "dynamic_threshold"
        "denoised_type": "dynamic_threshold", #("dynamic_threshold", "clip_denoised") только для "Kandinsky2.0"
        "model_type": "dpt_hybrid", #Только для модели "SD-2.0", тип модели ("dpt_large", "dpt_hybrid", "midas_v21", "midas_v21_small")
        "verbose": True, #Не знаю что это
        "s_churn": 0.0,  #Только для обработчиков "EulerEDMSampler" или "HeunEDMSampler" (от 0.0 до 1.0)
        "s_tmin": 0.0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы меньше этого значения (от 0.0 до "sigma_max" и < "s_tmax")
        "s_tmax": 999.0,  #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы больше этого значения (от "sigma_min" до "sigma_max" и > "s_tmin")
        "s_noise": 1.0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler" или "EulerAncestralSampler" или "DPMPP2SAncestralSampler") и "s_churn" > 0 (от 0.0)
        "eta": 1.0, #Только для обработчика "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
        "order": 4, #Только для обработчика "LinearMultistepSampler" (от 1)
        "force_i2i_resolution": False, #Если выбран, то размер итогового изображения для I2I генерации будет взят из параметров (w, h), а не у исходного изображения
        "i2i_strength": 0.75, #вклад в генерацию модели в режиме I2I (от 0.0 до 1.0)
        "i2i_prior_strength": 0.85, #Только для "Kandinsky2.2" с "ControlNET"
        "i2i_negative_prior_strength": 1.0, #Только для "Kandinsky2.2" с "ControlNET"
        "m_k": 8, #Коэффициент улучшения при постобработке (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "aesthetic_score": 6.0, #Эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "negative_aesthetic_score": 2.5, #Обратный эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "custom_orig_size": False, #Если применён, то меняет размеры входного изображения на "orig_width" и "orig_heigt", иначе оставляет равними размерам желаемого изображения
        "orig_width": 1024, #Ширина входного изображения, если установлен параметр "custom_orig_size" (от 16)
        "orig_heigt": 1024, #Высота входного изображения, если установлен параметр "custom_orig_size" (от 16)
        "crop_coords_top": 0, #Обрезка координат сверху (от 0)
        "crop_coords_left": 0, #Обрезка координат слева (от 0)
        "guider_discretization": "VanillaCFG", #Дискретизатор проводника? ("VanillaCFG", "IdentityGuider")
        "sampling_discretization": "LegacyDDPMDiscretization", #Дискретизатор обработчика ("LegacyDDPMDiscretization", "EDMDiscretization")
        "sigma_min": 0.03, #Только для "EDMDiscretization" дискритизатора обработчика
        "sigma_max": 14.61, #Только для "EDMDiscretization" дискритизатора обработчика
        "rho": 3.0, #Только для "EDMDiscretization" дискритизатора обработчика
        "num_cols": 1, #Количество возвращаемых изображений (от 1 до 10, но, думаю, можно и больше при желании)
        "guidance_scale": 9.0,
        "prior_scale": 4, #Только для Kandinsky > 2.0
        "steps": 40, #Количество шагов обработки (от 0 до 1000)
        "prior_steps": 25 #Только для Kandinsky > 2.0
    }
    '''
    if params["version"] == "SD-2.0":
        if params["Depth"] == True:
            binary_data_list = [Stable_diffusion_2_0_depth_to_image(init_img_binary_data, caption, params)]
        else:
            binary_data_list = [Stable_diffusion_2_0_image_to_image(init_img_binary_data, caption, params)]
    elif "Kandinsky" in params["version"]:
        binary_data_list = Kandinsky2_image_to_image(init_img_binary_data, caption, params)
    elif "webui_img2img" in params["version"]:
        binary_data_list = webui_img2img(init_img_binary_data, caption, params)
    else:
        binary_data_list = Stable_diffusion_XL_image_to_image(init_img_binary_data, caption, params)
    return binary_data_list

#Генерация по тексту (Text to image) (4 разные нейронки, у них разные параметры, в комментариях всё указано, требуется максимум внимания)
#Принимает описание и параметры. Возвращает список изображений
def text_to_image(caption: str, params: dict) -> List[bytes]:
    '''
    params = { 
        #Параметры не для пользователя:
        "add_watermark": False, #Добавлять невидимую вотермарку
        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
        #на данный момент "max_dim": 8 * pow(2048, 2) / f
        #Параметры для пользователя:
        "version": "SDXL-base-1.0", # Выбор модели: "SDXL-base-1.0", "SDXL-base-0.9", "Kandinsky2.0", "Kandinsky2.1", "Kandinsky2.2", (недоступна для коммерческого использования), "SD-2.0", "SD-2.1", "SD-2.1-768", "SDXL-refiner-0.9" (недоступна для коммерческого использования, используется как модель 2 стадии, для первой непригодна), "SDXL-refiner-1.0" (используется как модель 2 стадии, для первой непригодна)
        "ControlNET": False, #Только для "Kandinsky2.2"
        "use_flash_attention": False, #Только для "Kandinsky"
        "progress": True, #Только для Kandinsky < 2.2 и обработчика "p_sampler"
        "dynamic_threshold_v": 99.5, #Только для "Kandinsky2.0" и "dynamic_threshold"
        "denoised_type": "dynamic_threshold", #("dynamic_threshold", "clip_denoised") только для "Kandinsky2.0"
        "use_custom_ckpt": False, #Использовать свои веса для выбранной версии модели
        "custom_ckpt_name": "v2-1_512-ema-pruned.safetensors", #Имя кастомной модели, либо (если выбран "use_custom_ckpt", обязательный параметр), либо (для модели "SD-2.0", как обязательный параметр. Может быть "v2-1_512-ema-pruned.safetensors", "v2-1_512-nonema-pruned.safetensors", "v2-1_768-ema-pruned.safetensors", "v2-1_768-nonema-pruned.safetensors")      
        "seed": 42, #Инициализирующее значение (может быть от 0 до 1000000000)
        "negative_prompt": "", #Для всех моделей, кроме (SDXL-base и Kandinsky 2.0): негативное описание
        "negative_prior_prompt": "", #Только для Kandinsky > 2.0
        "refiner": "SDXL-refiner-1.0", #Если "version2SDXL-refiner" выбран, то какую версию модели для второй стадии обработки загрузить: "SDXL-refiner-1.0", "SDXL-refiner-0.9"  (недоступна для коммерческого использования)
        "refinement_strength": 0.15, #Сила вклада обработки на второй стадии (от 0.0 до 1.0)
        "finish_denoising": True, #Завершить удаление шума рафинёром (только для моделей SDXL-base, если включён version2SDXL-refiner)
        "h": 1024, #Высота желаемого изображения (от 64 до 2048, должна быть кратна 64) (512 для Kandinsky 2.0 и SD 512, 768 для Kandinsky 2.1 и SD 2.1 768 и 1024 для Kandinsky 2.2 и SDXL)
        "w": 1024, #Ширина желаемого изображения (от 64 до 2048, должна быть кратна 64) (512 для Kandinsky 2.0 и SD 512, 768 для Kandinsky 2.1 и SD 2.1 768 и 1024 для Kandinsky 2.2 и SDXL)
        "f": 8, #Коэффициент понижающей дискретизации, чаще всего 8 или 16 (можно 4, тогда есть риск учетверения, но красиво и жрёт больше видеопамяти) (От 4 до 64), если (4 можно, если w * h <= 1024 * 1024, иначе > 4)
        "use_recommended_res": True, #Использовать рекомендованное для каждой модели разрешение генерации, вместо указанных выше
        "sampler": "EulerEDMSampler", #Обработчик (Для SD > 2.0: "EulerEDMSampler", "HeunEDMSampler", "EulerAncestralSampler", "DPMPP2SAncestralSampler", "DPMPP2MSampler", "LinearMultistepSampler") и (Kandinsky < 2.2 или SD-2.0: "ddim_sampler", "plms_sampler", "p_sampler")
        "ddim_eta": 0.05, #только для обработчика "ddim_sampler" и (SD-2.0 или Kandinsky < 2.2)
        "s_churn": 0.0,  #Только для обработчиков "EulerEDMSampler" или "HeunEDMSampler" (от 0.0 до 1.0)
        "s_tmin": 0.0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы меньше этого значения (от 0.0 до "sigma_max" и < "s_tmax")
        "s_tmax": 999.0,  #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы больше этого значения (от "sigma_min" до "sigma_max" и > "s_tmin")
        "s_noise": 1.0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler" или "EulerAncestralSampler" или "DPMPP2SAncestralSampler") и "s_churn" > 0 (от 0.0)
        "eta": 1.0, #Только для обработчика "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
        "order": 4, #Только для обработчика "LinearMultistepSampler" (от 1)
        "m_k": 8, #Коэффициент улучшения при постобработке (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "aesthetic_score": 6.0, #Эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "negative_aesthetic_score": 2.5, #Обратный эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "custom_orig_size": False, #Если применён, то меняет размеры входного изображения на "orig_width" и "orig_heigt", иначе оставляет равними размерам желаемого изображения
        "orig_width": 1024, #Ширина входного изображения, если установлен параметр "custom_orig_size" (от 16)
        "orig_heigt": 1024, #Высота входного изображения, если установлен параметр "custom_orig_size" (от 16)
        "crop_coords_top": 0, #Обрезка координат сверху (от 0)
        "crop_coords_left": 0, #Обрезка координат слева (от 0)
        "guider_discretization": "VanillaCFG", #Дискретизатор проводника? ("VanillaCFG", "IdentityGuider")
        "sampling_discretization": "LegacyDDPMDiscretization", #Дискретизатор обработчика ("LegacyDDPMDiscretization", "EDMDiscretization")
        "sigma_min": 0.03, #Только для "EDMDiscretization" дискритизатора обработчика
        "sigma_max": 14.61, #Только для "EDMDiscretization" дискритизатора обработчика
        "rho": 3.0, #Только для "EDMDiscretization" дискритизатора обработчика
        "num_cols": 1, #Количество возвращаемых изображений (от 1 до 10, но, думаю, можно и больше при желании)
        "prior_scale": 4, #Только для Kandinsky > 2.0
        "guidance_scale": 5.0, #Величина guidance (от 0.0 до 100.0)
        "steps": 40, #Количество шагов обработки (от 0 до 1000)
        "prior_steps": 25, #Только для Kandinsky > 2.0
    }
    '''
    # binary_data_list = [b""]
    # '''
    # params["refinement_strength"] = 0.15
    # params["h"] = 256
    # params["w"] = 256
    # params["use_custom_res"] = True
    # params["version"] = "SDXL-base-1.0"
    if params["version"] == "SD-2.0":
        binary_data_list = [Stable_diffusion_2_0_text_to_image(caption, params)]
    #elif "Kandinsky" in params["version"]:
        #binary_data_list = Kandinsky2_text_to_image(caption, params)
    #elif "webui_text2img" in params["version"]:
        #binary_data_list = webui_text2img(caption, params)   
    else:
        binary_data_list = [webui_text2img(caption, params)]  
        #binary_data_list = Stable_diffusion_XL_text_to_image(caption, params)
        # with open("C:\\Users\\Robolightning\\Desktop\\cow.png", "wb") as f:
        #     f.write(binary_data_list[0])
    # '''
    # with open("C:\\Users\\Robolightning\\Desktop\\cow illustration in the meadow.png", "rb") as f:
        # binary_data_list = [f.read()]
    return binary_data_list

#Генерация описания для изображения (Image captioning) (1 нейронка, параметры только для неё. Но скоро будет заменена другой нейронкой)
#Принимает изображение и параметры. Возвращает строку описания
def image_captioning(init_img_binary_data: bytes, params: dict) -> str:
    '''
    params = {
        "ckpt": "caption_base_best.pt", #используемые чекпоинты (caption_huge_best.pt или caption_base_best.pt)
        "eval_cider": True,             #оценка с помощью баллов CIDEr
        "eval_bleu": False,             #оценка с помощью баллов BLEU
        "eval_args": "{}",              #аргументы генерации для оценки BLUE или CIDEr, например, "{"beam": 4, "lenpen": 0,6}", в виде строки JSON
        "eval_print_samples": False,    #печатать поколения образцов во время валидации
        "scst": False,                  #Обучение самокритичной последовательности
        "scst_args": "{}",              #аргументы генерации для обучения самокритичной последовательности в виде строки JSON
        "beam": 5,                      #балансировка
        "max_len_a": 0,                 #максимальная длина буфера a
        "max_len_b": 100,               #максимальная длина буфера b
        "min_len": 1,                   #минимальная длина буфера
        "unnormalized": False,          #ненормализовывать
        "lenpen": 1,
        "unkpen": 0,
        "temperature": 1.0,             #температура
        "match_source_len": False,      #сопоставлять с исходной длиной
        "no_repeat_ngram_size": 3,      #не повторять N-граммы размера
        "sampling_topk": 3,             #из скольки тоненов отбирать лучший (0 - не использовать сэмплирование)
        "seed": 42                      #инициализирующее значение для генерации
    }#max_dim не обнаружено. Возможно даунсемплит где-то внутри
    '''
    caption = Gen_caption(init_img_binary_data, params)
    return caption

#Определение категории изображения (Classification) (Техническая нейронка, необходимая только для внутреннего использования. 1 модель без входных параметров)
#Принимает изображение. Возвращает список из двух цифр: 1 - номер класса, 2 - номер подкласса
#Виды классов (по индексам): 
'''
classes = [
    "фото с лицом",
    "фото без лица",
    "профессиональный рисунок",
    "непрофессиональный рисунок",
    "профессиональный лайн",
    "быстрый лайн"
]
'''
#Виды подклассов (по индексам): 
'''
subclasses = [
    "в цвете",
    "чб"
]
'''
def image_classification(init_img_binary_data: bytes) -> List[int]:
    class_name = Get_image_class(init_img_binary_data)
    return class_name

#Перевод текста (Translate) (это даже не нейронка, параметров не принимает, просто переводит текст, пока по API, потом будет заменена нейронкой. Нужна исключительно для технических целей. но можно в будущем дать доступ к переводчику пользователям, при желании, в отдельной менюшке)
#Вначале нужно вызвать инициализатор:
translator = Translator()
#Принимает строку текста для перевода, опционально: (строку ключа языка, с которого осуществляется перевод, скажем "en", если не передавать, то определит язык источника автоматически. Если он будет совпадать с языком назвачения - вернёт строку буз перевода) и (строку ключа языка, на который осуществляется перевод, скажем "en", если не указывать, по умолчанию переводит на английский). Возвращает кортеж из строки ключа языка источника (по нему можно определить, был ли произведён перевод, или он равен ключу языка назвачения, и с какого языка перевод осуществлялся, если язык источника не был указан) и строки переведённого текста
def translation(input_text: str, source_lang: str = "", dest_lang: str = "en") -> tuple:
    final_source_lang, translated_text = translator.translate(input_text, source_lang, dest_lang)
    return (final_source_lang, translated_text)

#Перерисовка области изображения (Inpainting) (2 модели, нужно внимательно читать комментарии к параметра, чтобы понять что к чему относится)
#Принимает на вход исходное изображение, изображение маски (где содержимое маски может иметь любой цвет, но значение альфаканала равно 255, а область вне маски является прозрачное, то есть альфаканал в этих местах равен 0), строку описания и словарь параметров. Возвращает список изображений
def inpainting(init_img_binary_data: bytes, mask_binary_data: bytes, caption: str, params: dict) -> List[bytes]:
    '''
    params = {
        "version": "Kandinsky2.2",                  #"SD-2.0", "Kandinsky2.0", "Kandinsky2.1", "Kandinsky2.2"
        "progress": True,                           #Только для Kandinsky < 2.2 и обработчика "p_sampler"
        "dynamic_threshold_v": 99.5,                #Только для "Kandinsky2.0" и "dynamic_threshold"
        "denoised_type": "dynamic_threshold",       #("dynamic_threshold", "clip_denoised") только для "Kandinsky2.0"
        "negative_prior_prompt": "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature", #Только для Kandinsky > 2.0
        "negative_prompt": "",                      #Только для Kandinsky > 2.0
        "low_vram_mode": False,                     #Только для Kandinsky: режим для работы на малом количестве видеопамяти
        "sampler": "ddim_sampler",                  #("ddim_sampler", "plms_sampler", "p_sampler") для всех моделей, кроме Kandinsky 2.2
        "steps": 50,                                #Шаги от 0 до 50
        "prior_steps": 25,                          #Только для Kandinsky > 2.0
        "num_cols": 1,                              #Количество вариций изображений за одну генерацию
        "ddim_eta": 0.0,                            #значения от 0.0 до 1.0, η = 0.0 соответствует детерминированной выборке, только для обработчика "ddim_sampler" и для всех версий, кроме Kandinsky 2.2
        "guidance_scale": 10.0,                     #от 0.1 до 30.0
        "prior_scale": 4,                           #Только для Kandinsky > 2.0
        "strength": 0.9,                            #сила увеличения/уменьшения шума. 1.0 соответствует полному уничтожению информации в инициализирующем образе
        "ckpt": "512-inpainting-ema.safetensors",   #выбор весов модели ("512-inpainting-ema.safetensors")
        "seed": 42,                                 #от 0 до 1000000
        "verbose": False,                           #понятия не имею что это
        "max_dim": pow(2048, 2)                     #я не могу генерировать на своей видюхе картинки больше 2048 на 2048
    }
    '''
    if params["version"] == "SD-2.0":
        binary_data_list = [Stable_diffusion_inpainting(init_img_binary_data, mask_binary_data, caption, params)] #передаю сокет, путь к рабочей папке, имя файла и параметры
    else:
        binary_data_list = Kandinsky2_inpainting(init_img_binary_data, mask_binary_data, caption, params)
    return binary_data_list

#Стилизация (Style transfer) (1 модель пока что, скоро будет вторая)
#Принимает на вход изображение контента, изображение стиля, строку описания и словарь параметров. Возвращает список изображений
def stylization(content_binary_data: bytes, style_binary_data: bytes, caption: str, params: dict) -> List[bytes]:
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
    '''
    binary_data_list = Kandinsky2_stylization(content_binary_data, style_binary_data, caption, params)
    return binary_data_list

#Совмещение изображений (Image fusion) (1 модель, но у неё 2 версии, внимательно читайте комментарии к параметрам, они могу относиться к разным версиям)
#Принимает первое и второе изображения для совмещения, описание первого и второго изображения для совмещения и словарь параметров. Возвращает список изображений
def image_fusion(img1_binary_data: bytes, img2_binary_data: bytes, prompt1: str, prompt2: str, params: dict) -> List[bytes]:
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
    '''
    binary_data_list = Kandinsky2_mix_images(img1_binary_data, img2_binary_data,  prompt1, prompt2, params)
    return binary_data_list