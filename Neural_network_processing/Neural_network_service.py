import io
from PIL import Image
from huggingface_hub import cached_assets_path
from INeural import colorizer
from INeural import delete_background
from INeural import upscaler
from INeural import image_to_image
from INeural import text_to_image
from INeural import image_captioning
from INeural import image_classification
from INeural import translation
from INeural import inpainting
from INeural import stylization
from INeural import image_fusion

if __name__ == "__main__":
    
    caption = "vector logo of drone on brain background"
    with open("test_input\\img.png", "rb") as f:
        init_img_binary_data = f.read()
    with open("test_input\\img2.png", "rb") as f:
        mask_binary_data = f.read()
        
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
    
    binary_data = colorizer(init_img_binary_data, params)
    

    params = {
        "model": "DIS", #Доступно "U2NET" или "DIS"
        "RescaleT": 320, #Только для модели U2NET
        #Только для модели "DIS":
        "ckpt": "isnet.pth",        # Выбор впретренированных весов модели ("isnet.pth", "isnet-general-use.pth")
        "interm_sup": False,        # Указать, активировать ли контроль промежуточных функций
        "model_digit": "full",      # Выберите точность с плавающей запятой (устанавливает "half" или "full" точность числа с плавающей запятой)
        "seed": 0,                  # Инициализирующее значение
        "cache_size": [1024, 1024], # Кешированное входное пространственное разрешение, можно настроить на другой размер
        "input_size": [1024, 1024], # Входной пространственный размер модели, обычно используют одно и то же значение params["cache_size"], что означает, что мы больше не изменяем размер изображений
        "crop_size": [1024, 1024]   # Размер случайно обрезки из ввода, обычно он меньше, чем params["cache_size"], например, [920, 920] для увеличения данных
    #max_dim, возможно, даунсемплится по "*_size" параметрам выше. Не исследованно, и зависит от модели. По умолчанию, ограничений нет
    }
    
    binary_data = delete_background(init_img_binary_data, params)
    

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
        "negative_prompt": None,                #отрицательное описание (если без него, то None)
        "verbose": False,                       #Не знаю что это
        "max_dim": pow(1024, 2),                #Максимальное разрешение ((для всех моделей, кроме "RealESRGAN_x2plus") и "outscale": 4), и pow(2048, 2) (для модели "RealESRGAN_x2plus" и "outscale": 2)
        #Только для моделей RealESR:
        "denoise_strength": 0.5,                #Сила удаления шума. 0 для слабого удаления шума (шум сохраняется), 1 для сильного удаления шума. Используется только для модели "realesr-general-x4v3"
        "tile": 0,                              #Размер плитки, 0 для отсутствия плитки во время тестирования, влияет на количество требуемой видеопамяти и скорость обработки
        "tile_pad": 10,                         #Заполнение плитки
        "pre_pad": 0,                           #Предварительный размер заполнения на каждой границе
        "face_enhance": False,                  #Использовать GFPGAN улучшения лиц
        "fp32": True,                           #Использовать точность fp32 во время вывода. По умолчанию fp16 (половинная точность)
        "alpha_upsampler": "realesrgan",        #Апсемплер для альфа-каналов. Варианты: realesrgan | bicubic
        "gpu-id": None                          #Устройство gpu для использования (по умолчанию = None) может быть 0, 1, 2 для обработки на нескольких GPU
    }
    
    binary_data = upscaler(init_img_binary_data, caption, params)


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
    
    binary_data = image_to_image(init_img_binary_data, caption, params)[0]
    

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
        "version2SDXL-refiner": False, #Только для версий SDXL-base: загрузить SDXL-refiner как модель для второй стадии обработки. Требует более длительной обработки и больше видеопамяти
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

    binary_data = text_to_image(caption, params)[0]
    

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

    nn_caption = image_captioning(init_img_binary_data, params)
    print(nn_caption)


    classes = [
        "фото с лицом",
        "фото без лица",
        "профессиональный рисунок",
        "непрофессиональный рисунок",
        "профессиональный лайн",
        "быстрый лайн"
    ]

    #Виды подклассов (по индексам): 
    subclasses = [
        "в цвете",
        "чб"
    ]
    
    clss = image_classification(init_img_binary_data)
    print(subclasses[clss[1]] + " " + classes[clss[0]])


    nn_caption = translation(input_text = caption, source_lang = "en", dest_lang = "ru")
    print(nn_caption)
    

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
    
    binary_data = inpainting(init_img_binary_data, mask_binary_data, caption, params)[0]


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
    content_binary_data = init_img_binary_data
    style_binary_data = mask_binary_data
    
    binary_data = stylization(content_binary_data, style_binary_data, caption, params)[0]


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
    img1_binary_data = init_img_binary_data
    img2_binary_data = mask_binary_data
    caption1 = caption
    caption2 = "colorful logo of snake"

    binary_data = image_fusion(img1_binary_data, img2_binary_data, caption1, caption2, params)[0]


    output_image = Image.open(io.BytesIO(binary_data)).convert("RGBA")
    output_image.show()
    output_image.save("test_output\\img.png")