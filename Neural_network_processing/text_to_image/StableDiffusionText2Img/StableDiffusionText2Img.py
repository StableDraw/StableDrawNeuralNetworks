import io
import cv2
import torch
import numpy as np
import safetensors.torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

def load_model_from_config(config, ckpt, verbose = False):
    print(f"Загрузка модели из {ckpt}")
    if ckpt[ckpt.rfind('.'):] == ".safetensors":
        pl_sd = safetensors.torch.load_file(ckpt, device = "cpu")
    else:
        pl_sd = torch.load(ckpt, map_location = "cpu")
    if "global_step" in pl_sd:
        print(f"Глобальный шаг: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict = False)
    if len(m) > 0 and verbose:
        print("Пропущенные параметры:\n", m)
    if len(u) > 0 and verbose:
        print("Некорректные параматры:")
        print(u)
    model.cuda()
    model.eval()
    return model

def Stable_diffusion_text_to_image(prompt, opt):
    checkpoint_path = "weights\\"
    version_dict = {
        "2.0-512": ["512-base-ema.safetensors", "v1-inference.yaml"],
        "2.0-768-v": ["768-v-ema.safetensors", "v1-inference.yaml"],
        "2.1-512-ema": ["v2-1_512-ema-pruned.safetensors", "v2-inference.yaml"],
        "2.1-512-nonema": ["v2-1_512-nonema-pruned.safetensors", "v2-inference.yaml"],
        "2.1-768-ema": ["v2-1_768-ema-pruned.safetensors", "v2-inference.yaml"],
        "2.1-768-nonema": ["v2-1_768-nonema-pruned.safetensors", "v2-inference.yaml"]
        }
    config_path = "configs\\"
    if opt["use_custom_res"] == False:
        w = 512
        h = 512
        if "768" in opt["version"]:
            w = 768
            h = 768
    else:
        w, h = map(lambda x: x - x % 64, (opt["w"], opt["h"]))  # изменение размера в целое число, кратное 64-м
        if w == 0:
            w = 64
        if h == 0:
            h = 64
    f = 8
    h //= f
    w //= f
    seed_everything(opt["seed"])
    config = OmegaConf.load(config_path + version_dict[opt["version"]][1])
    model = load_model_from_config(config, checkpoint_path + version_dict[opt["version"]][0], verbose = opt["verbose"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if opt["sampler"] == "plms":
        sampler = PLMSSampler(model, device = device)
    elif opt["sampler"] == "dpm":
        sampler = DPMSolverSampler(model, device = device)
    else:
        sampler = DDIMSampler(model, device = device)
    if opt["add_watermark"]:
        wm = "SDV2"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark("bytes", wm.encode("utf-8"))

    batch_size = opt["num_samples"]
    assert prompt is not None
    data = [batch_size * [prompt]]

    start_code = torch.randn([opt["num_samples"], opt["C"], h, w], device=device)

    if opt["torchscript"] or opt["ipex"]:
        transformer = model.cond_stage_model.model
        unet = model.model.diffusion_model
        decoder = model.first_stage_model.decoder
        additional_context = torch.cpu.amp.autocast() if opt["bf16"] else nullcontext()
        shape = [opt["C"], h, w]

        if opt["bf16"] and not opt["torchscript"] and not opt["ipex"]:
            raise ValueError('Bfloat16 is supported only for torchscript+ipex')
        if opt["bf16"] and unet.dtype != torch.bfloat16:
            raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if you'd like to use bfloat16 with CPU.")
        if unet.dtype == torch.float16 and device == torch.device("cpu"):
            raise ValueError("Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

        if opt["ipex"]:
            import intel_extension_for_pytorch as ipex
            bf16_dtype = torch.bfloat16 if opt["bf16"] else None
            transformer = transformer.to(memory_format=torch.channels_last)
            transformer = ipex.optimize(transformer, level="O1", inplace=True)
            unet = unet.to(memory_format=torch.channels_last)
            unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)
            decoder = decoder.to(memory_format=torch.channels_last)
            decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

        if opt["torchscript"]:
            with torch.no_grad(), additional_context:
                # get UNET scripted
                if unet.use_checkpoint:
                    raise ValueError("Gradient checkpoint won't work with tracing. Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")
                img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                t_in = torch.ones(2, dtype = torch.int64)
                context = torch.ones(2, 77, 1024, dtype=torch.float32)
                scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
                print(type(scripted_unet))
                model.model.scripted_diffusion_model = scripted_unet
                # get Decoder for first stage model scripted
                samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
                print(type(scripted_decoder))
                model.first_stage_model.decoder = scripted_decoder
        print("Running a forward pass to initialize optimizations")
        uc = None
        if opt["scale"] != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        with torch.no_grad(), additional_context:
            for _ in range(3):
                c = model.get_learned_conditioning(prompts)
            samples_ddim, _ = sampler.sample(S = 5, conditioning = c, batch_size = batch_size, shape = shape, verbose = opt["verbose"], unconditional_guidance_scale = opt["scale"], unconditional_conditioning = uc, discretize = opt["discretize"], eta = opt["eta"], x_T = start_code)
            print("Running a forward pass for decoder")
            for _ in range(3):
                x_samples_ddim = model.decode_first_stage(samples_ddim)

    precision_scope = autocast if opt["precision"] == False or opt["bf16"] else nullcontext
    b_data_list = []
    with torch.no_grad(), precision_scope(device), model.ema_scope():
            for n in trange(opt["n_iter"], desc = "Sampling"):
                for prompts in tqdm(data, desc = "data"):
                    uc = None
                    if opt["scale"] != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt["C"], h, w]
                    samples, _ = sampler.sample(S=opt["steps"], conditioning = c, batch_size = opt["num_samples"], shape = shape, verbose = opt["verbose"], unconditional_guidance_scale = opt["scale"], unconditional_conditioning = uc, discretize = opt["discretize"], eta = opt["eta"], x_T = start_code)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if opt["add_watermark"]:
                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            if wm_encoder is not None:
                                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                img = wm_encoder.encode(img, 'dwtDct')
                                img = Image.fromarray(img[:, :, ::-1])
                            buf = io.BytesIO()
                            img.save(buf, format = "PNG")
                            b_data = buf.getvalue()
                            img.close
                            b_data_list.append(b_data)
                    else:
                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            buf = io.BytesIO()
                            img.save(buf, format = "PNG")
                            b_data = buf.getvalue()
                            img.close
                            b_data_list.append(b_data)
    torch.cuda.empty_cache()
    return b_data_list

if __name__ == "__main__":
    params = { 
        #Параметры для пользователя:
        "version": "2.1-512-ema", #Имя кастомной модели. Может быть "2.0-512", "2.0-768-v", "2.1-512-ema", "2.1-512-nonema", "2.1-768-ema", "2.1-768-nonema")   
        "steps": 40, #Количество шагов обработки (от 0 до 1000)
        "n_iter": 3, #Количество итераций?
        "C": 4, #Латентные каналы?
        "seed": 42, #Инициализирующее значение (может быть от 0 до 1000000000)
        "use_custom_res": False, #Использовать рекомендованное для каждой модели разрешение генерации, вместо указанных выше
        "h": 512, #Высота желаемого изображения (от 64 до 2048, должна быть кратна 64) (Только если выбран "use_custom_res": True)
        "w": 512, #Ширина желаемого изображения (от 64 до 2048, должна быть кратна 64) (Только если выбран "use_custom_res": True)
        "sampler": "ddim", #Обработчик "ddim", "plms", "dpm"
        "num_samples": 2, #Количество возвращаемых изображений (от 1 до 10, но, думаю, можно и больше при желании)
        "eta": 0.05, #только для обработчика "ddim_sampler" и SD-2.0
        "discretize": "uniform",    #Дискретизатор обработчика (доступны "uniform" и "quad"), только при eta > 0.0
        "scale": 9.0,                  #От 0.1 до 30.0
        "precision": True, #Вычислять с полной точностью? (если нет, то автоматически. Это быстрее, но менее качественно). Системный параметр
        "bf16": False, #Если выбран "prescision": True, то использовать половинную точность. Иначе -- автоматически. Это системный параметр
        "torchscript": False, #Использовать torchscript. Системный параметр
        "ipex": False, #Использовать оптисизацию для Intel (На данный момент у меня AMD). Системный параметр
        "add_watermark": False,     #Добавлять невидимую вотермарку
        "verbose": False,
        "max_dim": 1048576  #Hа данный момент "max_dim": pow(1024, 2)
    }

    prompt = "a professional photograph of an astronaut riding a triceratops"
    binary_data_list = Stable_diffusion_text_to_image(prompt, params)
    i = 0
    for binary_data in binary_data_list:
        Image.open(io.BytesIO(binary_data)).save("img" + str(i) + ".png")
        i += 1