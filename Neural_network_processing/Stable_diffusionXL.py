from pytorch_lightning import seed_everything
import math
from typing import List, Union
import numpy as np
import torch
from einops import rearrange, repeat
from imwatermark import WatermarkEncoder
from omegaconf import ListConfig, OmegaConf
import io
import PIL
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch import autocast
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.modules.diffusionmodules.sampling import (DPMPP2MSampler, DPMPP2SAncestralSampler, EulerAncestralSampler, EulerEDMSampler, HeunEDMSampler, LinearMultistepSampler)
from sgm.util import append_dims, instantiate_from_config

def load_model_from_config(config, ckpt, verbose = False):
    print(f"Загрузка модели из {ckpt}")
    if ckpt[ckpt.rfind('.'):] == ".safetensors":
        pl_sd = load_safetensors(ckpt, device = "cpu")
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

def load_img(binary_data, max_dim):
    image = PIL.Image.open(io.BytesIO(binary_data)).convert("RGB")
    orig_w, orig_h = image.size
    print(f"Загружено входное изображение размера ({orig_w}, {orig_h})")
    cur_dim = orig_w * orig_h
    if cur_dim > max_dim:
        k = cur_dim / max_dim
        sk = float(k ** (0.5))
        w, h = int(orig_w / sk), int(orig_h / sk)
    else:
        w, h = orig_w, orig_h
    w, h = map(lambda x: x - x % 64, (w, h))  # изменение размера в целое число, кратное 64-м
    if w == 0 and orig_w != 0:
        w = 64
    if h == 0 and orig_h != 0:
        h = 64
    if (w, h) != (orig_w, orig_h):
        image = image.resize((w, h), resample = PIL.Image.LANCZOS)
        print(f"Размер изображения изменён на ({w}, {h} (w, h))")
    else:
        print(f"Размер исходного изображения не был изменён")
    return image

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    },
    "SDXL-base-0.9": {
        "H": 1024,
        "W": 1024,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_0.9.safetensors",
    },
    "SD-2.1": {
        "H": 512,
        "W": 512,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/v2-1_512-ema-pruned.safetensors",
    },
    "SD-2.1-768": {
        "H": 768,
        "W": 768,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1_768.yaml",
        "ckpt": "checkpoints/v2-1_768-ema-pruned.safetensors",
    },
    "SDXL-refiner-0.9": {
        "H": 1024,
        "W": 1024,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_0.9.safetensors",
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_1.0.safetensors",
    },
}

class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)
        
    def __call__(self, image: torch.Tensor):
        """
        Добавляет предопределенную вотермарку к входному изображению

        Args:
            image: ([N,] B, C, H, W) in range [0, 1]

        Returns:
            то же, что и на вход, но с вотермаркой
        """
        # Вотермарочная библиотека ожидает вход, как cv2 BGR формат
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(image.device)
        image = torch.clamp(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        return image

# Исправленное 48-битное сообщение, которое было рандомно выбрано
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] принимает x как str, использует int, чтобы сконвертировать его в 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watemark = WatermarkEmbedder(WATERMARK_BITS)

def init_sd(opt, version_dict, load_ckpt = True, load_filter = True):
    state = dict()
    if not "model" in state:
        config = version_dict["config"]
        if opt["use_custom_ckpt"] == True:
            ckpt = opt["custom_ckpt_name"]
        else:
            ckpt = version_dict["ckpt"]
        config = OmegaConf.load(config)
        model = load_model_from_config(config, ckpt if load_ckpt else None)
        state["model"] = model
        state["ckpt"] = ckpt if load_ckpt else None
        state["config"] = config
        if load_filter:
            state["filter"] = DeepFloydDataFiltering(verbose = False)
    return state

lowvram_mode = False

def set_lowvram_mode(mode):
    global lowvram_mode
    lowvram_mode = mode

def initial_model_load(model):
    global lowvram_mode
    if lowvram_mode:
        model.model.half()
    else:
        model.cuda()
    return model

def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def init_embedder_options(opt, keys, init_dict, prompt = None, negative_prompt = None):
    value_dict = {}
    for key in keys:
        if key == "txt":
            if negative_prompt is None:
                negative_prompt = ""
            value_dict["prompt"] = prompt
            value_dict["negative_prompt"] = negative_prompt
        if key == "original_size_as_tuple":
            if opt["custom_orig_width"] == True:
                value_dict["orig_width"] = opt["orig_width"]
                value_dict["orig_height"] = opt["orig_height"]
            else:
                value_dict["orig_width"] = init_dict["orig_width"]
                value_dict["orig_height"] = init_dict["orig_height"]
        if key == "crop_coords_top_left":
            value_dict["crop_coords_top"] = opt["crop_coords_top"]
            value_dict["crop_coords_left"] = opt["crop_coords_left"]
        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = opt["aesthetic_score"]
            value_dict["negative_aesthetic_score"] = opt["negative_aesthetic_score"]
        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]
    return value_dict

class Img2ImgDiscretizationWrapper:
    """
    обертывает дискретизатор и обрезает сигмы
    params:
        сила вклада обработки: float между 0.0 и 1.0. 1.0 означает полное семплирование (возвращаются все сигмы)
    """
    def __init__(self, discretization, strength: float = 1.0):
        self.discretization = discretization
        self.strength = strength
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        #сигмы сначала начинаются большими, а потом уменьшаются
        sigmas = self.discretization(*args, **kwargs)
        print(f"Сигмы после дискретизации, до обрезки img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        sigmas = sigmas[: max(int(self.strength * len(sigmas)), 1)]
        print("Индекс обрезки: ", max(int(self.strength * len(sigmas)), 1))
        sigmas = torch.flip(sigmas, (0,))
        print(f"Сигмы после обрезки: ", sigmas)
        return sigmas

class Txt2NoisyDiscretizationWrapper:
    """
    обертывает дискретизатор и обрезает сигмы
    params:
        сила вклада: float между 0.0 и 1.0. 0.0 означает полное семплирование (возвращаются все сигмы)
    """
    def __init__(self, discretization, strength: float = 0.0, original_steps=None):
        self.discretization = discretization
        self.strength = strength
        self.original_steps = original_steps
        assert 0.0 <= self.strength <= 1.0

    def __call__(self, *args, **kwargs):
        #сигмы сначала начинаются большими, а потом уменьшаются
        sigmas = self.discretization(*args, **kwargs)
        print(f"Сигмы после дискретизации, до обрезки img2img: ", sigmas)
        sigmas = torch.flip(sigmas, (0,))
        if self.original_steps is None:
            steps = len(sigmas)
        else:
            steps = self.original_steps + 1
        prune_index = max(min(int(self.strength * steps) - 1, steps - 1), 0)
        sigmas = sigmas[prune_index:]
        print("Индекс обрезки: ", prune_index)
        sigmas = torch.flip(sigmas, (0,))
        print(f"Сигмы после обрезки: ", sigmas)
        return sigmas

def get_guider(opt):
    guider = opt["guider_discretization"]
    if guider == "IdentityGuider":
        guider_config = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}
    elif guider == "VanillaCFG":
        scale = opt["cfg-scale"]
        dyn_thresh_config = {"target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"}
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": scale, "dyn_thresh_config": dyn_thresh_config},
        }
    return guider_config

def init_sampling(opt, img2img_strength = 1.0, specify_num_samples = True, stage2strength = None):
    num_rows, num_cols = 1, 1
    if specify_num_samples:
        num_cols = opt["num_cols"]
    discretization = opt["sampling_discretization"]
    discretization_config = get_discretization(opt, discretization)
    guider_config = get_guider(opt)
    sampler = get_sampler(opt, discretization_config, guider_config)
    if img2img_strength < 1.0:
        print(f"Предупреждение: {sampler.__class__.__name__} с Img2ImgDiscretizationWrapper")
        sampler.discretization = Img2ImgDiscretizationWrapper(sampler.discretization, strength = img2img_strength)
    if stage2strength is not None:
        sampler.discretization = Txt2NoisyDiscretizationWrapper(sampler.discretization, strength = stage2strength, original_steps = opt["steps"])
    return sampler, num_rows, num_cols


def get_discretization(opt, discretization):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {"target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"}
    elif discretization == "EDMDiscretization":
        sigma_min = opt["sigma_min"]
        sigma_max = opt["sigma_max"]
        rho = opt["rho"]
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            }
        }
    return discretization_config

def get_sampler(opt, discretization_config, guider_config):
    sampler_name = opt["sampler"]
    steps = opt["steps"]
    s_churn = (opt["s_churn"] * steps * (2**0.5 - 1))
    if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":
        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, s_churn = s_churn, s_tmin = opt["s_tmin"], s_tmax = opt["s_tmax"], s_noise = opt["s_noise"], verbose = True)
        elif sampler_name == "HeunEDMSampler":
            sampler = HeunEDMSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, s_churn = s_churn, s_tmin = opt["s_tmin"], s_tmax = opt["s_tmax"], s_noise = opt["s_noise"], verbose = True)
    elif (sampler_name == "EulerAncestralSampler" or sampler_name == "DPMPP2SAncestralSampler"):
        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, eta = opt["eta"], s_noise = opt["s_noise"], verbose = True)
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, eta = opt["eta"], s_noise = opt["s_noise"], verbose = True)
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, verbose = True)
    elif sampler_name == "LinearMultistepSampler":
        order = opt["order"]
        sampler = LinearMultistepSampler(num_steps = steps, discretization_config = discretization_config, guider_config = guider_config, order = order, verbose = True)
    return sampler

def do_sample(model, sampler, value_dict, num_samples, H, W, C, F, force_uc_zero_embeddings: List = None, batch2model_input: List = None, return_latents = False, filter = None):
    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    if batch2model_input is None:
        batch2model_input = []
    print("Обработка")
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                num_samples = [num_samples]
                model.conditioner.cuda()
                batch, batch_uc = get_batch(get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, num_samples)
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(key, batch[key].shape)
                    elif isinstance(batch[key], list):
                        print(key, [len(l) for l in batch[key]])
                    else:
                        print(key, batch[key])
                c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc = batch_uc, force_uc_zero_embeddings = force_uc_zero_embeddings)
                unload_model(model.conditioner)
                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc))
                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]
                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")
                def denoiser(input, sigma, c):
                    return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)
                model.denoiser.cuda()
                model.model.cuda()
                samples_z = sampler(denoiser, randn, cond = c, uc = uc)
                unload_model(model.model)
                unload_model(model.denoiser)
                model.first_stage_model.cuda()
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(model.first_stage_model)
                if filter is not None:
                    samples = filter(samples)
                if return_latents:
                    return samples, samples_z
                return samples

def get_batch(keys, value_dict, N: Union[List, ListConfig], device = "cuda"):
    # Захардкоженные демонстрационные пресеты
    batch = {}
    batch_uc = {}
    for key in keys:
        if key == "txt":
            batch["txt"] = (np.repeat([value_dict["prompt"]], repeats = math.prod(N)).reshape(N).tolist())
            batch_uc["txt"] = (np.repeat([value_dict["negative_prompt"]], repeats = math.prod(N)).reshape(N).tolist())
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (torch.tensor([value_dict["orig_height"], value_dict["orig_width"]]).to(device).repeat(*N, 1))
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (torch.tensor([value_dict["crop_coords_top"], value_dict["crop_coords_left"]]).to(device).repeat(*N, 1))
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1))
            batch_uc["aesthetic_score"] = (torch.tensor([value_dict["negative_aesthetic_score"]]).to(device).repeat(*N, 1))
        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (torch.tensor([value_dict["target_height"], value_dict["target_width"]]).to(device).repeat(*N, 1))
        else:
            batch[key] = value_dict[key]
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

@torch.no_grad()
def do_img2img(img, model, sampler, value_dict, num_samples, force_uc_zero_embeddings = [], additional_kwargs = {}, offset_noise_level: int = 0.0, return_latents = False, skip_encode = False, filter = None, add_noise = True):
    print("Обработка")
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                model.conditioner.cuda()
                batch, batch_uc = get_batch(get_unique_embedder_keys_from_conditioner(model.conditioner), value_dict, [num_samples])
                c, uc = model.conditioner.get_unconditional_conditioning(batch, batch_uc = batch_uc, force_uc_zero_embeddings = force_uc_zero_embeddings)
                unload_model(model.conditioner)
                for k in c:
                    c[k], uc[k] = map(lambda y: y[k][:num_samples].to("cuda"), (c, uc))
                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]
                if skip_encode:
                    z = img
                else:
                    model.first_stage_model.cuda()
                    z = model.encode_first_stage(img)
                    unload_model(model.first_stage_model)
                noise = torch.randn_like(z)
                sigmas = sampler.discretization(sampler.num_steps).cuda()
                sigma = sigmas[0]
                print(f"Все сигмы: {sigmas}")
                print(f"Шумовая сигма: {sigma}")
                if offset_noise_level > 0.0:
                    noise = noise + offset_noise_level * append_dims(torch.randn(z.shape[0], device=z.device), z.ndim)
                if add_noise:
                    noised_z = z + noise * append_dims(sigma, z.ndim).cuda()
                    noised_z = noised_z / torch.sqrt(1.0 + sigmas[0] ** 2.0)  # Заметка: захардкожено в DDPM-подобное преобразование. 
                else:
                    noised_z = z / torch.sqrt(1.0 + sigmas[0] ** 2.0)
                def denoiser(x, sigma, c):
                    return model.denoiser(model.model, x, sigma, c)
                model.denoiser.cuda()
                model.model.cuda()
                samples_z = sampler(denoiser, noised_z, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)
                model.first_stage_model.cuda()
                samples_x = model.decode_first_stage(samples_z)
                unload_model(model.first_stage_model)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                if filter is not None:
                    samples = filter(samples)
                if return_latents:
                    return samples, samples_z
                return samples

def prepare_SDXL(opt):
    version_dict = VERSION2SPECS[opt["version"]]
    set_lowvram_mode(opt["low_vram_mode"])
    if opt["version"].startswith("SDXL-base"):
        add_pipeline = opt["version2SDXL-refiner"]
    else:
        add_pipeline = False
    seed_everything(opt["seed"])
    state = init_sd(opt = opt, version_dict = version_dict, load_filter = True)
    is_legacy = version_dict["is_legacy"]
    if is_legacy:
        negative_prompt = opt["negative_prompt"]
    else:
        negative_prompt = ""  #оно не используется
    stage2strength = None
    finish_denoising = False
    state2 = None
    sampler2 = None
    if add_pipeline:
        state2 = init_sd(opt = opt, version_dict = VERSION2SPECS[opt["refiner"]], load_filter = False)
        stage2strength = opt["refinement_strength"]
        sampler2, *_ = init_sampling(opt = opt, img2img_strength = stage2strength, specify_num_samples = False)
        finish_denoising = opt["finish_denoising"]
        if not finish_denoising:
            stage2strength = None
    return state, add_pipeline, stage2strength, state2, sampler2, finish_denoising, negative_prompt

def apply_refiner(opt, input, state, sampler, num_samples, prompt, filter = None, finish_denoising = False):
    version_dict = VERSION2SPECS[opt["version"]]
    init_dict = {"orig_width": input.shape[3] * opt["m_k"], "orig_height": input.shape[2] * opt["m_k"], "target_width": input.shape[3] * opt["m_k"], "target_height": input.shape[2] * opt["m_k"]}
    value_dict = init_dict
    value_dict["prompt"] = prompt
    value_dict["negative_prompt"] = opt["negative_prompt"] if version_dict["is_legacy"] else ""
    value_dict["crop_coords_top"] = opt["crop_coords_top"]
    value_dict["crop_coords_left"] = opt["crop_coords_left"]
    value_dict["aesthetic_score"] = opt["aesthetic_score"]
    value_dict["negative_aesthetic_score"] = opt["negative_aesthetic_score"]
    print(f"Пропорции входного изображения рефайнера: {input.shape}")
    samples = do_img2img(input, state["model"], sampler, value_dict, num_samples, skip_encode = True, filter = filter, add_noise = not finish_denoising)
    return samples

def SDXL_postprocessing(opt, prompt, state, finish_denoising, out, add_pipeline, state2, sampler2):
    if isinstance(out, (tuple, list)):
        samples, samples_z = out
    else:
        samples = out
        samples_z = None
    if add_pipeline and samples_z is not None:
        print("Запуск этапа уточнения")
        samples = apply_refiner(opt, samples_z, state2, sampler2, samples_z.shape[0], prompt = prompt, filter = state.get("filter"), finish_denoising = finish_denoising)
    if opt["add_watermark"] == True:
        samples = embed_watemark(samples)
    r = []
    for sample in samples:
        sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
        image = Image.fromarray(sample.astype(np.uint8))
        buf = io.BytesIO()
        image.save(buf, format = "PNG")
        b_data = buf.getvalue()
        image.close
        r.append(b_data)
    torch.cuda.empty_cache()
    return r

def Stable_diffusion_XL_text_to_image(prompt, opt):
    state, return_latents, stage2strength, state2, sampler2, finish_denoising, negative_prompt = prepare_SDXL(opt)
    filter = state.get("filter")
    version_dict = VERSION2SPECS[opt["version"]]
    if opt["use_recommended_res"] == True:
        H = version_dict["H"]
        W = version_dict["W"]
    else:
        H = opt["h"]
        W = opt["w"]
    C = 4
    F = opt["f"]
    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(opt, get_unique_embedder_keys_from_conditioner(state["model"].conditioner), init_dict, prompt = prompt, negative_prompt = negative_prompt if version_dict["is_legacy"] else "")
    sampler, num_rows, num_cols = init_sampling(opt = opt, stage2strength = stage2strength)
    num_samples = num_rows * num_cols
    out = do_sample(state["model"], sampler, value_dict, num_samples, H, W, C, F, force_uc_zero_embeddings = ["txt"] if not version_dict["is_legacy"] else [], return_latents = return_latents, filter = filter)
    return SDXL_postprocessing(opt, prompt, state, finish_denoising, out, return_latents, state2, sampler2)

def Stable_diffusion_XL_image_to_image(binary_data, prompt, opt):
    state, return_latents, stage2strength, state2, sampler2, finish_denoising, negative_prompt = prepare_SDXL(opt)
    filter = state.get("filter")
    version_dict = VERSION2SPECS[opt["version"]]
    image = load_img(binary_data, opt["max_dim"])
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype = torch.float32) / 127.5 - 1.0
    img = image.to("cuda")
    H, W = img.shape[2], img.shape[3]
    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    if opt["force_i2i_resolution"] == True:
        init_dict["target_width"] == opt["w"]
        init_dict["target_height"] == opt["h"]
    value_dict = init_embedder_options(opt, get_unique_embedder_keys_from_conditioner(state["model"].conditioner), init_dict, prompt = prompt, negative_prompt = negative_prompt if version_dict["is_legacy"] else "")
    strength = opt["i2i_strength"]
    sampler, num_rows, num_cols = init_sampling(opt = opt, img2img_strength = strength, stage2strength = stage2strength)
    num_samples = num_rows * num_cols
    out = do_img2img(repeat(img, "1 ... -> n ...", n = num_samples), state["model"], sampler, value_dict, num_samples, force_uc_zero_embeddings = ["txt"] if not version_dict["is_legacy"] else [], return_latents = return_latents, filter = filter)
    return SDXL_postprocessing(opt, prompt, state, finish_denoising, out, return_latents, state2, sampler2)
'''
if __name__ == "__main__":

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k", #Описание
    opt = {
        "add_watermark": False, #Добавлять невидимую вотермарку
        "version": "SDXL-refiner-1.0", # Выбор модели: "SDXL-base-1.0", "SDXL-base-0.9" (недоступна для коммерческого использования), "SD-2.1", "SD-2.1-768", "SDXL-refiner-0.9" (недоступна для коммерческого использования),  "SDXL-refiner-1.0"
        "use_custom_ckpt": False, #Использовать свои веса для выбранной версии модели
        "custom_ckpt_name": "sd_xl_refiner_1.0.safetensors", #Имя кастомной модели, если выбран "use_custom_ckpt"
        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
        "version2SDXL-refiner": True, #Только для версий SDXL-base: загрузить SDXL-refiner как модель для второй стадии обработки. Требует более длительной обработки и больше видеопамяти
        "seed": 42, #Инициализирующее значение (может быть от 0 до 1000000000)
        "negative_prompt": "", #Для всех моделей, кроме SDXL-base: негативное описание
        "refiner": "SDXL-refiner-1.0", #Если "version2SDXL-refiner" выбран, то какую версию модели для второй стадии обработки загрузить: "SDXL-refiner-1.0", "SDXL-refiner-0.9"  (недоступна для коммерческого использования)
        "refinement_strength": 0.15, #Сила вклада обработки на второй стадии (от 0.0 до 1.0)
        "finish_denoising": True, #Завершить удаление шума рафинёром (только для моделей SDXL-base, если включён version2SDXL-refiner)
        "h": 1024, #Высота желаемого изображения (от 64 до 2048, должна быть кратна 64)
        "w": 1024, #Ширина желаемого изображения (от 64 до 2048, должна быть кратна 64)
        "max_dim": pow(8192, 2), # я не могу генерировать на своей видюхе картинки больше x на x (проверить эксперементальным путём)
        "c": 4, #Кто знает, что это -- напишите
        "f": 8, #Коэффициент понижающей дискретизации, чаще всего 8 или 16 (можно 4, тогда есть риск учетверения, но красиво и жрёт больше видеопамяти)
        "use_recommended_res": True, #Использовать рекомендованное для каждой модели разрешение генерации, вместо указанных выше
        "sampler": "EulerEDMSampler", #обработчик ("EulerEDMSampler", "HeunEDMSampler", "EulerAncestralSampler", "DPMPP2SAncestralSampler", "DPMPP2MSampler", "LinearMultistepSampler")
        "s_churn": 0.0,  #Только для обработчика "EulerEDMSampler" или "HeunEDMSampler" (от 0.0)
        "s_tmin": 0.0, #Только для обработчика "EulerEDMSampler" или "HeunEDMSampler" (от 0.0)
        "s_tmax": 999.0, #Только для обработчика "EulerEDMSampler" или "HeunEDMSampler" (от 0.0)
        "s_noise": 1.0, #Только для обработчика "EulerEDMSampler" или "HeunEDMSampler" или "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
        "eta": 1.0, #Только для обработчика "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
        "order": 4, #Только для обработчика "LinearMultistepSampler" (от 1)
        "m_k": 8, #Коэффициент улучшения при постобработке (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "aesthetic_score": 6.0, #Эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "negative_aesthetic_score": 2.5, #Обратный эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "custom_orig_width": False, #Если применён, то меняет размеры входного изображения на "orig_width" и "orig_heigt", иначе оставляет равними размерам желаемого изображения
        "orig_width": 1024, #Ширина входного изображения, если установлен параметр "custom_orig_width" (от 16)
        "orig_heigt": 1024, #Высота входного изображения, если установлен параметр "custom_orig_width" (от 16)
        "crop_coords_top": 0, #Обрезка координат сверху (от 0)
        "crop_coords_left": 0, #Обрезка координат слева (от 0)
        "guider_discretization": "VanillaCFG", #Дискретизатор проводника? ("VanillaCFG", "IdentityGuider")
        "sampling_discretization": "LegacyDDPMDiscretization", #Дискретизатор обработчика ("LegacyDDPMDiscretization", "EDMDiscretization")
        "sigma_min": 0.03, #Только для "EDMDiscretization" дискритизатора обработчика
        "sigma_max": 14.61, #Только для "EDMDiscretization" дискритизатора обработчика
        "rho": 3.0, #Только для "EDMDiscretization" дискритизатора обработчика
        "num_cols": 2, #Количество столбцов? Не знаю что это (от 1 до 10)
        "cfg-scale": 5.0, #Размер cfg (от 0.0 до 100.0)
        "steps": 40, #Количество шагов обработки (от 0 до 1000)
    }
    r = Stable_diffusion_XL_text_to_image(prompt, opt)
    c = 0
    for rr in r:
        binary_data = rr
        c += 1
        with open("output_" + str(c) + ".png", "wb") as f:
            f.write(binary_data)

    opt = {
        "add_watermark": False, #Добавлять невидимую вотермарку
        "version": "SDXL-refiner-1.0", # Выбор модели: "SDXL-base-1.0", "SDXL-base-0.9" (недоступна для коммерческого использования), "SD-2.1", "SD-2.1-768", "SDXL-refiner-0.9" (недоступна для коммерческого использования),  "SDXL-refiner-1.0"
        "use_custom_ckpt": False, #Использовать свои веса для выбранной версии модели
        "custom_ckpt_name": "sd_xl_refiner_1.0.safetensors", #Имя кастомной модели, если выбран "use_custom_ckpt"
        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
        "mode": "txt2img", #Выбор режима: "txt2img", "img2img"
        "version2SDXL-refiner": True, #Только для версий SDXL-base: загрузить SDXL-refiner как модель для второй стадии обработки. Требует более длительной обработки и больше видеопамяти
        "seed": 42, #Инициализирующее значение (может быть от 0 до 1000000000)
        "negative_prompt": "", #Для всех моделей, кроме SDXL-base: негативное описание
        "refiner": "SDXL-refiner-1.0", #Если "version2SDXL-refiner" выбран, то какую версию модели для второй стадии обработки загрузить: "SDXL-refiner-1.0", "SDXL-refiner-0.9"  (недоступна для коммерческого использования)
        "refinement_strength": 0.15, #Сила вклада обработки на второй стадии (от 0.0 до 1.0)
        "finish_denoising": True, #Завершить удаление шума рафинёром (только для моделей SDXL-base, если включён version2SDXL-refiner)
        "h": 1024, #Высота желаемого изображения (от 64 до 2048, должна быть кратна 64)
        "w": 1024, #Ширина желаемого изображения (от 64 до 2048, должна быть кратна 64)
        "max_dim": pow(8192, 2), # я не могу генерировать на своей видюхе картинки больше x на x (проверить эксперементальным путём)
        "c": 4, #Кто знает, что это -- напишите
        "f": 8, #коэффициент понижающей дискретизации, чаще всего 8 или 16 (можно 4, тогда есть риск учетверения, но красиво и жрёт больше видеопамяти)
        "use_recommended_res": True, #Использовать рекомендованное для каждой модели разрешение генерации, вместо указанных выше
        "sampler": "EulerEDMSampler", #обработчик ("EulerEDMSampler", "HeunEDMSampler", "EulerAncestralSampler", "DPMPP2SAncestralSampler", "DPMPP2MSampler", "LinearMultistepSampler")
        "s_churn": 0.0,  #Только для обработчика "EulerEDMSampler" или "HeunEDMSampler" (от 0.0)
        "s_tmin": 0.0, #Только для обработчика "EulerEDMSampler" или "HeunEDMSampler" (от 0.0)
        "s_tmax": 999.0, #Только для обработчика "EulerEDMSampler" или "HeunEDMSampler" (от 0.0)
        "s_noise": 1.0, #Только для обработчика "EulerEDMSampler" или "HeunEDMSampler" или "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
        "eta": 1.0, #Только для обработчика "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
        "order": 4, #Только для обработчика "LinearMultistepSampler" (от 1)
        "force_i2i_resolution": False, #Если выбран, то размер итогового изображения для I2I генерации будет взят из параметров (w, h), а не у исходного изображения
        "i2i_strength": 0.75, #вклад в генерацию модели в режиме I2I (от 0.0 до 1.0)
        "m_k": 8, #Коэффициент улучшения при постобработке (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "aesthetic_score": 6.0, #Эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "negative_aesthetic_score": 2.5, #Обратный эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
        "custom_orig_width": False, #Если применён, то меняет размеры входного изображения на "orig_width" и "orig_heigt", иначе оставляет равними размерам желаемого изображения
        "orig_width": 1024, #Ширина входного изображения, если установлен параметр "custom_orig_width" (от 16)
        "orig_heigt": 1024, #Высота входного изображения, если установлен параметр "custom_orig_width" (от 16)
        "crop_coords_top": 0, #Обрезка координат сверху (от 0)
        "crop_coords_left": 0, #Обрезка координат слева (от 0)
        "guider_discretization": "VanillaCFG", #Дискретизатор проводника? ("VanillaCFG", "IdentityGuider")
        "sampling_discretization": "LegacyDDPMDiscretization", #Дискретизатор обработчика ("LegacyDDPMDiscretization", "EDMDiscretization")
        "sigma_min": 0.03, #Только для "EDMDiscretization" дискритизатора обработчика
        "sigma_max": 14.61, #Только для "EDMDiscretization" дискритизатора обработчика
        "rho": 3.0, #Только для "EDMDiscretization" дискритизатора обработчика
        "num_cols": 2, #Количество столбцов? Не знаю что это (от 1 до 10)
        "cfg-scale": 5.0, #Размер cfg (от 0.0 до 100.0)
        "steps": 40, #Количество шагов обработки (от 0 до 1000)
    }
    with open("input.jpg", "rb") as f:
        binary_data = f.read()    
    r = Stable_diffusion_XL_image_to_image(binary_data, prompt, opt)
    c = 0
    for rr in r:
        binary_data = rr
        c += 1
        with open("output_" + str(c) + ".png", "wb") as f:
            f.write(binary_data)
'''