import io
import PIL
import torch
import numpy
import safetensors.torch
from omegaconf import OmegaConf
from torch import autocast
from itertools import islice
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from ldm.util import exists, instantiate_from_config
from ldm.data.util import AddMiDaS
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentUpscaleFinetuneDiffusion
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import is_accelerate_available, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

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

def Stable_diffusion_text_to_image(prompt, opt):
    torch.set_grad_enabled(False)
    checkpoint_path = "models\\ldm\\stable-diffusion\\text2img\\"
    checkpoint_list = {
        "v2-1_512-ema-pruned.safetensors": 1,
        "v2-1_512-nonema-pruned.safetensors": 1,
        "v2-1_768-ema-pruned.safetensors": 1,
        "v2-1_768-nonema-pruned.safetensors": 1,
        "cornflowerStylizedAnime_v8.safetensors": 0,
    }
    config_path = "configs\\stable-diffusion\\"
    config_list = ["v1-inference.yaml", "v2-inference.yaml"]
    w = 512
    h = 512
    if ("1_768" in opt["custom_ckpt_name"]):
        w = 768
        h = 768
    seed_everything(opt["seed"])
    config = OmegaConf.load(config_path + config_list[checkpoint_list[opt["custom_ckpt_name"]]])
    model = load_model_from_config(config, checkpoint_path + opt["custom_ckpt_name"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if opt["sampler"] == "plms_sampler":
        sampler = PLMSSampler(model)
    elif opt["sampler"] == "p_sampler":
        sampler = DPMSolverSampler(model)
        opt["steps"] += 1
    else:
        sampler = DDIMSampler(model)
    precision_scope = autocast
    with torch.no_grad(), \
        precision_scope("cuda"), \
        model.ema_scope():
            if opt["guidance_scale"] != 1.0:
                uc = model.get_learned_conditioning([""])
            else:
                uc = None
            c = model.get_learned_conditioning([prompt])
            shape = [4, h // opt["f"], w // opt["f"]]
            samples, _ = sampler.sample(S = opt["steps"], conditioning = c, batch_size = 1, shape = shape, verbose = False, unconditional_guidance_scale = opt["guidance_scale"], unconditional_conditioning = uc, eta = opt["ddim_eta"], x_T = None)
            x_sample = 255. * rearrange(torch.clamp((model.decode_first_stage(samples) + 1.0) / 2.0, min = 0.0, max = 1.0)[0].cpu().numpy(), 'c h w -> h w c')
            img = PIL.Image.fromarray(x_sample.astype(numpy.uint8))
            buf = io.BytesIO()
            img.save(buf, format = "PNG")
            b_data = buf.getvalue()
            img.close
    print("Обработка успешно завершена")
    torch.cuda.empty_cache()
    return b_data

def Stable_diffusion_image_to_image(binary_data, prompt, opt):
    checkpoint_path = 'models\\ldm\\stable-diffusion\\img2img\\'
    checkpoint_list = {
        "sd-v1-1.safetensors": 0,
        "sd-v1-1-full-ema.safetensors": 0,
        "sd-v1-2.safetensors": 0,
        "sd-v1-2-full-ema.safetensors": 0,
        "sd-v1-3.safetensors": 0,
        "sd-v1-3-full-ema.safetensors": 0,
        "sd-v1-4.safetensors": 0,
        "sd-v1-4-full-ema.safetensors": 0,
        "sd-v1-5.safetensors": 0,
        "sd-v1-5-full-ema.safetensors": 0,
    }
    config_path = "configs\\stable-diffusion\\"
    config_list = ["v1-inference.yaml", "v2-inference.yaml"]
    seed_everything(opt["seed"])
    config = OmegaConf.load(config_path + config_list[checkpoint_list[opt["custom_ckpt_name"]]])
    model = load_model_from_config(config, checkpoint_path + opt["custom_ckpt_name"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if opt["sampler"] == "plms_sampler":
        sampler = PLMSSampler(model)
    elif opt["sampler"] == "p_sampler":
        sampler = DPMSolverSampler(model)
        opt["steps"] += 1
    elif opt["sampler"] == "ddim_sampler":
        sampler = DDIMSampler(model)
    else:
        raise ValueError("Only ddim_sampler and plms_sampler and p_sampler is available")
    init_image = repeat(torch.from_numpy(2.0 * (numpy.array(load_img(binary_data, opt["max_dim"])).astype(numpy.float32) / 255.0)[None].transpose(0, 3, 1, 2) - 1.).to(device), '1 ... -> b ...', b = 1)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # переместить в латентное пространство
    sampler.make_schedule(ddim_num_steps = opt["steps"], ddim_eta = opt["ddim_eta"], verbose = False)
    assert 0. <= opt["i2i_strength"] <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt["i2i_strength"] * opt["steps"])
    print(f"Целевое декодирование t_enc из {t_enc} шагов")
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                if opt["guidance_scale"] != 1.0:
                    uc = model.get_learned_conditioning([""])
                else:
                    uc = None
                c = model.get_learned_conditioning(prompt)
                # закодировать (скрытое масштабирование)
                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
                # раскодировать
                samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale = opt["guidance_scale"], unconditional_conditioning = uc, )
                x_sample = 255. * rearrange(torch.clamp((model.decode_first_stage(samples) + 1.0) / 2.0, min = 0.0, max = 1.0)[0].cpu().numpy(), 'c h w -> h w c')
                img = PIL.Image.fromarray(x_sample.astype(numpy.uint8))
                buf = io.BytesIO()
                img.save(buf, format = "PNG")
                b_data = buf.getvalue()
                img.close
    torch.cuda.empty_cache()
    return b_data

def Stable_diffusion_depth_to_image(binary_data, prompt, opt):
    torch.set_grad_enabled(False)
    checkpoint_path = "models\\ldm\\stable-diffusion\\dept2img\\"
    checkpoint_list = {
        "512-depth-ema.safetensors": 0
    }
    config_path = "configs\\stable-diffusion\\"
    config_list = ["v2-midas-inference.yaml"]
    config = config_path + config_list[checkpoint_list[opt["custom_ckpt_name"]]]
    seed_everything(opt["seed"])
    image = load_img(binary_data, opt["max_dim"])
    assert 0. <= opt["i2i_strength"] <= 1., "может работать с параметром шума в интервале от 0.0 до 1.0"
    if opt["i2i_strength"] == 1.:
        do_full_sample = True
    else:
        do_full_sample = False
    t_enc = min(int(opt["i2i_strength"] * opt["steps"]), opt["steps"] - 1)
    config = OmegaConf.load(config_path + config_list[checkpoint_list[opt["custom_ckpt_name"]]])
    model = load_model_from_config(config, checkpoint_path + opt["custom_ckpt_name"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if opt["sampler"] == "plms_sampler":
        sampler = PLMSSampler(model)
    elif opt["sampler"] == "p_sampler":
        sampler = DPMSolverSampler(model)
        opt["steps"] += 1
    elif opt["sampler"] == "ddim_sampler":
        sampler = DDIMSampler(model)
    else:
        raise ValueError("Only ddim_sampler and plms_sampler and p_sampler is available")
    sampler.make_schedule(ddim_num_steps = opt["steps"], ddim_eta = opt["ddim_eta"], verbose = opt["verbose"])
    image = numpy.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype = torch.float32) / 127.5 - 1.0
    # sample["jpg"] это тензор hwc в [-1, 1] в этом месте
    midas_trafo = AddMiDaS(model_type = opt["model_type"])
    batch = {
        "jpg": image,
        "txt": [prompt],
    }
    batch = midas_trafo(batch)
    batch["jpg"] = repeat(rearrange(batch["jpg"], 'h w c -> 1 c h w').to(device = device), "1 ... -> n ...", n = 1)
    batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(device = device), "1 ... -> n ...", n = 1)
    with torch.no_grad(), torch.autocast("cuda"):
        z = model.get_first_stage_encoding(model.encode_first_stage(batch[model.first_stage_key]))  # move to latent space
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        cc = model.depth_model(batch["midas_in"])
        depth_min, depth_max = torch.amin(cc, dim = [1, 2, 3], keepdim = True), torch.amax(cc, dim = [1, 2, 3], keepdim = True)
        cc = torch.nn.functional.interpolate(cc, size = z.shape[2:], mode = "bicubic", align_corners = False, )
        depth_min, depth_max = torch.amin(cc, dim = [1, 2, 3], keepdim = True), torch.amax(cc, dim = [1, 2, 3], keepdim = True)
        cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
        c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim = 1)
        # условие
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}
        # безусловное условие
        uc_cross = model.get_unconditional_conditioning(1, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        if not do_full_sample:
            # закодировать (латентное изменение размера)
            z_enc = sampler.stochastic_encode(z, torch.tensor([t_enc]).to(model.device))
        else:
            z_enc = torch.randn_like(z)
        # декодирование
        samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale = opt["guidance_scale"], unconditional_conditioning = uc_full)
        x_sample = 255. * rearrange(torch.clamp((model.decode_first_stage(samples) + 1.0) / 2.0, min = 0.0, max = 1.0)[0].cpu().numpy(), 'c h w -> h w c')
    image = PIL.Image.fromarray(x_sample.astype(numpy.uint8))
    buf = io.BytesIO()
    image.save(buf, format = "PNG")
    b_data = buf.getvalue()
    image.close
    torch.cuda.empty_cache()
    return b_data

def Stable_diffusion_inpainting(binary_data, mask_data, prompt, opt):
    torch.set_grad_enabled(False)
    checkpoint_path = 'models\\ldm\\stable-diffusion\\inpainting\\'
    checkpoint_list = {
        "512-inpainting-ema.safetensors": 1,
        "galaxytimemachines_v3.safetensors": 0,
    }
    config_path = "configs\\stable-diffusion\\"
    config_list = ["v1-inpainting-inference.yaml", "v2-inpainting-inference.yaml"]
    config = OmegaConf.load(config_path + config_list[checkpoint_list[opt["ckpt"]]])
    model = load_model_from_config(config, checkpoint_path + opt["ckpt"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if opt["sampler"] == "plms_sampler":
        sampler = PLMSSampler(model)
    elif opt["sampler"] == "p_sampler":
        sampler = DPMSolverSampler(model)
        opt["steps"] += 1
    else:
        sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps = opt['ddim_steps'], ddim_eta = opt['ddim_eta'], verbose = opt["verbose"])
    image = load_img(binary_data, opt["max_dim"])
    w, h = image.size
    mask = PIL.Image.open(io.BytesIO(mask_data)).resize((w, h))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    prng = numpy.random.RandomState(opt["seed"])
    start_code = prng.randn(1, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device = device, dtype = torch.float32)
    with torch.no_grad(), torch.autocast("cuda"):
            image = numpy.array(image.convert("RGB"))
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype = torch.float32) / 127.5 - 1.0
            mask = numpy.array(mask.convert("L"))
            mask = mask.astype(numpy.float32) / 255.0
            mask = mask[None, None]
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)
            masked_image = image * (mask < 0.5)
            batch = {
                "image": repeat(image.to(device = device), "1 ... -> n ...", n = 1),
                "txt": [prompt],
                "mask": repeat(mask.to(device = device), "1 ... -> n ...", n = 1),
                "masked_image": repeat(masked_image.to(device = device), "1 ... -> n ...", n = 1),
            }
            c = model.cond_stage_model.encode([prompt])
            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [1, 4, h // 8, w // 8]
                    cc = torch.nn.functional.interpolate(cc, size = bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim = 1)
            # услолвие
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # безусловное условие
            uc_cross = model.get_unconditional_conditioning(1, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
            shape = [model.channels, h // 8, w // 8]
            samples_cfg = sampler.sample(opt["steps"], 1, shape, cond, opt["verbose"], eta = opt["ddim_eta"], unconditional_guidance_scale = opt["guidance_scale"], unconditional_conditioning=uc_full, x_T=start_code, )[0]
            x_samples_ddim = model.decode_first_stage(samples_cfg)
            result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min = 0.0, max = 1.0).cpu().numpy().transpose(0, 2, 3, 1) * 255
    image = [PIL.Image.fromarray(img.astype(numpy.uint8)) for img in result][0]
    buf = io.BytesIO()
    image.save(buf, format = "PNG")
    b_data = buf.getvalue()
    image.close
    torch.cuda.empty_cache()
    return b_data

def Stable_diffusion_upscaler(binary_data, prompt, opt):
    w, h = PIL.Image.open(io.BytesIO(binary_data)).convert("RGB").size
    if opt["outscale"] != 4 or w * h > pow(opt["max_dim"], 2):
        return Stable_diffusion_upscaler_xX(binary_data, prompt, opt)
    torch.set_grad_enabled(False)
    checkpoint_path = "models\\ldm\\stable-diffusion\\upscaler\\"
    checkpoint_list = {
        "x4-upscaler-ema.safetensors": 0
    }
    config_path = "configs\\stable-diffusion\\"
    config_list = ["x4-upscaling.yaml"]
    image = load_img(binary_data, opt["max_dim"])
    w, h = image.size
    config = OmegaConf.load(config_path + config_list[0])
    model = load_model_from_config(config, checkpoint_path + opt["ckpt"], opt["verbose"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if opt["sampler"] == "plms_sampler":
        sampler = PLMSSampler(model)
    elif opt["sampler"] == "p_sampler":
        sampler = DPMSolverSampler(model)
        opt["steps"] += 1
    else:
        sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps = opt["steps"], ddim_discretize = opt["ddim_discretize"], ddim_eta = opt["ddim_eta"], verbose = opt["verbose"])
    if isinstance(sampler.model, LatentUpscaleDiffusion):
        noise_level = torch.Tensor([opt["noise_augmentation"]]).to(sampler.model.device).long()
    sampler.make_schedule(opt["steps"], ddim_eta = opt["ddim_eta"], verbose = opt["verbose"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(opt["seed"])
    prng = numpy.random.RandomState(opt["seed"])
    start_code = prng.randn(1, model.channels, h , w)
    start_code = torch.from_numpy(start_code).to(device = device, dtype = torch.float32)
    with torch.no_grad(), torch.autocast("cuda"):
        image = numpy.array(image.convert("RGB"))
        image = torch.from_numpy(image).to(dtype = torch.float32) / 127.5 - 1.0
        batch = {
            "lr": rearrange(image, 'h w c -> 1 c h w'),
            "txt": [prompt],
        }
        batch["lr"] = repeat(batch["lr"].to(device = device), "1 ... -> n ...", n = 1)
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        if isinstance(model, LatentUpscaleFinetuneDiffusion):
            for ck in model.concat_keys:
                cc = batch[ck]
                if exists(model.reshuffle_patch_size):
                    assert isinstance(model.reshuffle_patch_size, int)
                    cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w', p1 = model.reshuffle_patch_size, p2 = model.reshuffle_patch_size)
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim = 1)
            # условие
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # безусловное условие
            uc_cross = model.get_unconditional_conditioning(1, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        elif isinstance(model, LatentUpscaleDiffusion):
            x_low = batch[model.low_scale_key]
            x_low = x_low.to(memory_format = torch.contiguous_format).float()
            x_augment, noise_level = model.low_scale_model(x_low, noise_level)
            cond = {"c_concat": [x_augment], "c_crossattn": [c], "c_adm": noise_level}
            # безусловное условие
            uc_cross = model.get_unconditional_conditioning(1, "")
            uc_full = {"c_concat": [x_augment], "c_crossattn": [uc_cross], "c_adm": noise_level}
        else:
            raise NotImplementedError()
        shape = [model.channels, h, w]
        samples = sampler.sample(opt["steps"], 1, shape, cond, verbose = opt["verbose"], eta = opt["ddim_eta"], unconditional_guidance_scale = opt["guidance_scale"], unconditional_conditioning = uc_full, x_T = start_code, )[0]
    with torch.no_grad():
        x_samples_ddim = model.decode_first_stage(samples)
    result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min = 0.0, max = 1.0)
    result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    print(f"размер апскейленого изображения: {result.shape}")
    image = [PIL.Image.fromarray(img.astype(numpy.uint8)) for img in result][0]
    buf = io.BytesIO()
    image.save(buf, format = "PNG")
    b_data = buf.getvalue()
    image.close
    torch.cuda.empty_cache()
    return b_data

class StableDiffusionLatentUpscalePipeline(DiffusionPipeline):
    def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: EulerDiscreteScheduler,):
        super().__init__()
        self.register_modules(vae = vae, text_encoder = text_encoder, tokenizer = tokenizer, unet = unet, scheduler = scheduler)

    def enable_sequential_cpu_offload(self, gpu_id = 0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Пожалуйста, установите accelerate при помощи 'pip install accelerate'")
        device = torch.device(f"cuda:{gpu_id}")
        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device") and module._hf_hook.execution_device is not None):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.tokenizer(prompt, padding = "max_length", max_length = self.tokenizer.model_max_length, truncation = True, return_length = True, return_tensors = "pt")
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding = "longest", return_tensors = "pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            print(f"Следующая часть вашего ввода была усечена, потому что CLIP может обрабатывать последовательности только до {self.tokenizer.model_max_length} токенов: {removed_text}")
        text_encoder_out = self.text_encoder(text_input_ids.to(device), output_hidden_states = True)
        text_embeddings = text_encoder_out.hidden_states[-1]
        text_pooler_out = text_encoder_out.pooler_output
        # Получить безусловные эмбединги для классификации свободного управления
        if do_classifier_free_guidance:
            if negative_prompt is "":
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(f"'negative_prompt': {negative_prompt} имеет размер батча {len(negative_prompt)}, но 'описание': {prompt} размер батча {batch_size}. Пожалуйста убедитесь, что 'negative_prompt' соответствует размеру батча 'prompt'")
            else:
                uncond_tokens = negative_prompt
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(uncond_tokens, padding = "max_length", max_length = max_length, truncation = True, return_length = True, return_tensors = "pt")
            uncond_encoder_out = self.text_encoder(uncond_input.input_ids.to(device), output_hidden_states = True)
            uncond_embeddings = uncond_encoder_out.hidden_states[-1]
            uncond_pooler_out = uncond_encoder_out.pooler_output
            # Для классификации свободного управления нужно сделать два прямых прохода. Здесь мы объединяем безусловные и текстовые встраивания в один пакет, чтобы избежать двух прямых проходов
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            text_pooler_out = torch.cat([uncond_pooler_out, text_pooler_out])
        return text_embeddings, text_pooler_out

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # Мы всегда приводим к float32, поскольку это не вызывает значительных накладных расходов и совместимо с bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height, width)
        if latents is None:
            latents = randn_tensor(shape, generator = generator, device=device, dtype = dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Нераспознанная латентная форма, получено {latents.shape}, ожидалось {shape}")
            latents = latents.to(device)
        # Масштабировать начальный шум по стандартному отклонению, требуемому планировщиком
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(self, prompt, binary_data, opt, generator = None, latents = None, output_type = "pil", return_dict = True, callback = None, callback_steps = 1):
        # 1. Определение вызываемых параметров
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # Здесь "guidance_scale" определяется аналогично весу наведения "w" в уравнении (2) соответствует отсутствию свободного наведения классификатора
        guidance_scale = opt["guidance_scale"]
        do_classifier_free_guidance = guidance_scale > 1.0
        if guidance_scale == 0:
            prompt = [""] * batch_size
        # 2. Кодирование входного описания
        if opt["negative_prompt"] == "":
            negative_prompt = None
        else:
            negative_prompt = opt["negative_prompt"]
        text_embeddings, text_pooler_out = self._encode_prompt(prompt, device, do_classifier_free_guidance, negative_prompt)
        # 3. Обработка изображения
        image = torch.from_numpy(2.0 * (numpy.array(load_img(binary_data, opt["max_dim"])).astype(numpy.float32) / 255.0)[None].transpose(0, 3, 1, 2) - 1.)
        image = image.to(dtype = text_embeddings.dtype, device = device)
        if image.shape[1] == 3:
            # Кодировать изображение, если оно еще не находится в латентном пространстве
            image = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        # 4. Установка временных шагов
        num_inference_steps = opt["steps"]
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = image[None, :] if image.ndim == 3 else image
        image = torch.cat([image] * batch_multiplier)
        # 5. Добавление шума для изображения. Этот шаг теоретически может улучшить работу модели на входных данных вне распределения, но в основном он просто заставляет ее меньше соответствовать входным данным
        noise_level = torch.tensor([opt["noise_augmentation"]], dtype = torch.float32, device = device)
        noise_level = torch.cat([noise_level] * image.shape[0])
        inv_noise_level = (noise_level ** 2 + 1) ** (-0.5)
        image_cond = torch.nn.functional.interpolate(image, scale_factor = opt["outscale"], mode = "nearest") * inv_noise_level[:, None, None, None]
        image_cond = image_cond.to(text_embeddings.dtype)
        noise_level_embed = torch.cat(
            [
                torch.ones(text_pooler_out.shape[0], 64, dtype = text_pooler_out.dtype, device = device),
                torch.zeros(text_pooler_out.shape[0], 64, dtype = text_pooler_out.dtype, device = device),
            ],
            dim = 1,
        )
        timestep_condition = torch.cat([noise_level_embed, text_pooler_out], dim = 1)
        # 6. Подготавливаются латентные переменные
        height, width = image.shape[2:]
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(batch_size, num_channels_latents, height * opt["outscale"], width * opt["outscale"], text_embeddings.dtype, device, generator, latents)
        # 7. Проверка того, что размеры изображения и латенты совпадают
        num_channels_image = image.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(f"Некорректные настройки конфигурации! Конфигурация 'pipeline.unet': {self.unet.config} ожидалось {self.unet.config.in_channels} но получено 'num_channels_latents': {num_channels_latents} + 'num_channels_image': {num_channels_image} = {num_channels_latents + num_channels_image}. Пожалуйста сверьте конфигурацию 'pipeline.unet' или входной параметр 'image'")
        # 9. Цикл шумоподавления
        num_warmup_steps = 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                sigma = self.scheduler.sigmas[i]
                # Расширить латенты, если производится классификация свободного исполнения
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                scaled_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                scaled_model_input = torch.cat([scaled_model_input, image_cond], dim = 1)
                # Параметр предварительного кондиционирования, основанный на Karras полностью
                timestep = torch.log(sigma) * 0.25
                noise_pred = self.unet(scaled_model_input, timestep, encoder_hidden_states = text_embeddings, timestep_cond = timestep_condition).sample
                # В исходном репозитории выходные данные содержат неиспользуемый канал дисперсии
                noise_pred = noise_pred[:, :-1]
                # Применить предварительное кондиционирование на основе таблицы 1 в Karras полностью
                inv_sigma = 1 / (sigma ** 2 + 1)
                noise_pred = inv_sigma * latent_model_input + self.scheduler.scale_model_input(sigma, t) * noise_pred
                # Осуществлять управление
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # Вычисление предыдущей шумовой выборки x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                # Вызов обратного вызова, если он предоставлен
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        # 10. Постобработка
        image = self.decode_latents(latents)
        # 11. Конвертация в PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        if not return_dict:
            return (image,)
        image = image[0]
        buf = io.BytesIO()
        image.save(buf, format = "PNG")
        b_data = buf.getvalue()
        image.close
        torch.cuda.empty_cache()
        return b_data

def Stable_diffusion_upscaler_xX(init_img_binary_data, caption, params):
    upscaler_pipeline = StableDiffusionLatentUpscalePipeline
    upscaler = upscaler_pipeline.from_pretrained("models", torch_dtype = torch.float16)
    upscaler.to("cuda")
    prompt = caption
    generator = torch.manual_seed(params["seed"])
    return upscaler(prompt = prompt, binary_data = init_img_binary_data, generator = generator, opt = params)