import torch
from local_fastai.vision import *
from local_fastai.vision.learner import cnn_config
from local_fastai.vision.data import ImageImageList, ImageDataBunch
from local_fastai.callbacks.hooks import model_sizes, hook_outputs, dummy_eval
from tqdm import tqdm
from abc import ABC
import cv2

def custom_conv_layer(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None, bias: bool = None, is_1d: bool = False, norm_type = NormType.Batch, use_activ: bool = True, leaky: float = None, transpose: bool = False, init = nn.init.kaiming_normal_, self_attention: bool = False, extra_bn: bool = False):
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn == True
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if norm_type == NormType.Weight:
        conv = weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)

__all__ = ['DynamicUnetDeep', 'DynamicUnetWide']

def _get_sfs_idxs(sizes: Sizes) -> List[int]:
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]:
        sfs_idxs = [0] + sfs_idxs
    return sfs_idxs

class CustomPixelShuffle_ICNR(nn.Module):
    def __init__(self, ni: int, nf: int = None, scale: int = 2, blur: bool = False, leaky: float = None, **kwargs):
        super().__init__()
        nf = ifnone(nf, ni)
        self.conv = custom_conv_layer(ni, nf * (scale ** 2), ks=1, use_activ=False, **kwargs)
        icnr(self.conv[0].weight)
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x

class UnetBlockDeep(nn.Module):
    def __init__(self, up_in_c: int, x_in_c: int, hook, final_div: bool = True, blur: bool = False, leaky: float = None, self_attention: bool = False, nf_factor: float = 1.0, **kwargs):
        super().__init__()
        self.hook = hook
        self.shuf = CustomPixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, leaky=leaky, **kwargs)
        self.bn = batchnorm_2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = int((ni if final_div else ni // 2) * nf_factor)
        self.conv1 = custom_conv_layer(ni, nf, leaky=leaky, **kwargs)
        self.conv2 = custom_conv_layer(nf, nf, leaky=leaky, self_attention=self_attention, **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))

class DynamicUnetDeep(SequentialEx):
    def __init__(self, encoder: nn.Module, n_classes: int, blur: bool = False, blur_final=True, self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None, last_cross: bool = True,
        bottle: bool = False,
        norm_type: Optional[NormType] = NormType.Batch, nf_factor: float = 1.0, **kwargs):
        extra_bn = norm_type == NormType.Spectral
        imsize = (256, 256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(custom_conv_layer(ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs), custom_conv_layer(ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs)).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            sa = self_attention and (i == len(sfs_idxs) - 3)
            unet_block = UnetBlockDeep(up_in_c, x_in_c, self.sfs[i], final_div=not_final, blur=blur, self_attention=sa, norm_type=norm_type, extra_bn=extra_bn, nf_factor=nf_factor, **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs))
        layers += [custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type)]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()

class UnetBlockWide(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(
        self, up_in_c: int, x_in_c: int, n_out: int, hook, final_div: bool = True, blur: bool = False, leaky: float = None, self_attention: bool = False, **kwargs):
        super().__init__()
        self.hook = hook
        up_out = x_out = n_out // 2
        self.shuf = CustomPixelShuffle_ICNR(up_in_c, up_out, blur=blur, leaky=leaky, **kwargs)
        self.bn = batchnorm_2d(x_in_c)
        ni = up_out + x_in_c
        self.conv = custom_conv_layer(ni, x_out, leaky=leaky, self_attention=self_attention, **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor) -> Tensor:
        s = self.hook.stored
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv(cat_x)


class DynamicUnetWide(SequentialEx):
    def __init__(self, encoder: nn.Module, n_classes: int, blur: bool = False, blur_final=True, self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None, last_cross: bool = True, bottle: bool = False, norm_type: Optional[NormType] = NormType.Batch, nf_factor: int = 1, **kwargs):

        nf = 512 * nf_factor
        extra_bn = norm_type == NormType.Spectral
        imsize = (256, 256)
        sfs_szs = model_sizes(encoder, size=imsize)
        sfs_idxs = list(reversed(_get_sfs_idxs(sfs_szs)))
        self.sfs = hook_outputs([encoder[i] for i in sfs_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sfs_szs[-1][1]
        middle_conv = nn.Sequential(custom_conv_layer(ni, ni * 2, norm_type=norm_type, extra_bn=extra_bn, **kwargs), custom_conv_layer(ni * 2, ni, norm_type=norm_type, extra_bn=extra_bn, **kwargs),).eval()
        x = middle_conv(x)
        layers = [encoder, batchnorm_2d(ni), nn.ReLU(), middle_conv]

        for i, idx in enumerate(sfs_idxs):
            not_final = i != len(sfs_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sfs_szs[idx][1])
            sa = self_attention and (i == len(sfs_idxs) - 3)

            n_out = nf if not_final else nf // 2

            unet_block = UnetBlockWide(up_in_c, x_in_c, n_out, self.sfs[i], final_div=not_final, blur=blur, self_attention=sa, norm_type=norm_type, extra_bn=extra_bn, **kwargs).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sfs_szs[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, **kwargs))
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += in_channels(encoder)
            layers.append(res_block(ni, bottle=bottle, norm_type=norm_type, **kwargs))
        layers += [custom_conv_layer(ni, n_classes, ks=1, use_activ=False, norm_type=norm_type)]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()

def get_colorize_data(sz: int, bs: int, crappy_path: Path, good_path: Path, random_seed: int = None, keep_pct: float = 1.0, num_workers: int = 8, stats: tuple = imagenet_stats, xtra_tfms=[]) -> ImageDataBunch:
    src = (ImageImageList.from_folder(crappy_path, convert_mode='RGB').use_partial_data(sample_pct=keep_pct, seed=random_seed).split_by_rand_pct(0.1, seed=random_seed))
    data = (src.label_from_func(lambda x: good_path / x.relative_to(crappy_path)).transform(get_transforms(max_zoom=1.2, max_lighting=0.5, max_warp=0.25, xtra_tfms=xtra_tfms),size=sz, tfm_y=True).databunch(bs=bs, num_workers=num_workers, no_check=True).normalize(stats, do_y=True))
    data.c = 3
    return data


def get_dummy_databunch(stats=imagenet_stats) -> ImageDataBunch:
    path = Path('./dummy/')
    return get_colorize_data(sz=1, bs=1, crappy_path=path, good_path=path, stats=stats,keep_pct=0.001)

def gen_inference_wide(root_folder: Path, weights_name: str, nf_factor: int = 2, arch=models.resnet101, stats: tuple = imagenet_stats) -> Learner:
    data = get_dummy_databunch(stats)
    learn = gen_learner_wide(data=data, gen_loss=F.l1_loss, nf_factor=nf_factor, arch=arch)
    learn.path = root_folder
    learn.load(weights_name)
    learn.model.eval()
    return learn


def gen_learner_wide(data: ImageDataBunch, gen_loss, arch=models.resnet101, nf_factor: int = 2) -> Learner:
    return unet_learner_wide(data, arch=arch, wd=1e-3, blur=True, norm_type=NormType.Spectral, self_attention=True, y_range=(-3.0, 3.0), loss_func=gen_loss, nf_factor=nf_factor)

def unet_learner_wide(data: DataBunch,
    arch: Callable,
    pretrained: bool = True,
    blur_final: bool = True,
    norm_type: Optional[NormType] = NormType, split_on: Optional[SplitFuncOrIdxList] = None, blur: bool = False, self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None, last_cross: bool = True, bottle: bool = False, nf_factor: int = 1, **kwargs: Any) -> Learner:
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)
    model = to_device(DynamicUnetWide(body, n_classes=data.c, blur=blur, blur_final=blur_final, self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross, bottle=bottle, nf_factor=nf_factor), data.device)
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained:
        learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn

def gen_inference_deep(root_folder: Path, weights_name: str, arch=models.resnet34, nf_factor: float = 1.5, stats: tuple = imagenet_stats) -> Learner:
    data = get_dummy_databunch(stats=stats)
    learn = gen_learner_deep(data=data, gen_loss=F.l1_loss, arch=arch, nf_factor=nf_factor)
    learn.path = root_folder
    learn.load(weights_name)
    learn.model.eval()
    return learn


def gen_learner_deep(data: ImageDataBunch, gen_loss, arch=models.resnet34, nf_factor: float = 1.5) -> Learner:
    return unet_learner_deep(data, arch, wd=1e-3, blur=True, norm_type=NormType.Spectral, self_attention=True, y_range=(-3.0, 3.0), loss_func=gen_loss, nf_factor=nf_factor)

def unet_learner_deep(data: DataBunch, arch: Callable, pretrained: bool = True, blur_final: bool = True, norm_type: Optional[NormType] = NormType, split_on: Optional[SplitFuncOrIdxList] = None, blur: bool = False, self_attention: bool = False, y_range: Optional[Tuple[float, float]] = None, last_cross: bool = True, bottle: bool = False, nf_factor: float = 1.5, **kwargs: Any) -> Learner:
    meta = cnn_config(arch)
    body = create_body(arch, pretrained)
    model = to_device(DynamicUnetDeep(body, n_classes=data.c, blur=blur, blur_final=blur_final, self_attention=self_attention, y_range=y_range, norm_type=norm_type, last_cross=last_cross, bottle=bottle, nf_factor=nf_factor), data.device)
    learn = Learner(data, model, **kwargs)
    learn.split(ifnone(split_on, meta['split']))
    if pretrained:
        learn.freeze()
    apply_init(model[2], nn.init.kaiming_normal_)
    return learn

class IFilter(ABC):
    @abstractmethod
    def filter(self, orig_image, filtered_image, render_factor: int):
        pass

class BaseFilter(IFilter):
    def __init__(self, learn: Learner, stats:tuple = imagenet_stats):
        super().__init__()
        self.learn = learn
        self.norm, self.denorm = normalize_funcs(*stats)

    def _transform(self, image):
        return image

    def _scale_to_square(self, orig, targ: int):
        targ_sz = (targ, targ)
        return orig.resize(targ_sz, resample=PIL.Image.Resampling.BILINEAR)

    def _get_model_ready_image(self, orig, sz: int):
        result = self._scale_to_square(orig, sz)
        result = self._transform(result)
        return result

    def _model_process(self, orig, sz: int):
        model_image = self._get_model_ready_image(orig, sz)
        x = pil2tensor(model_image, np.float32)
        x.div_(255)
        x, y = self.norm((x, x), do_x=True)
        result = self.learn.pred_batch(ds_type=DatasetType.Valid, batch=(x[None].cuda(), y[None]), reconstruct=True)
        out = result[0]
        out = self.denorm(out.px, do_x=False)
        out = image2np(out * 255).astype(np.uint8)
        return PIL.Image.fromarray(out)

    def _unsquare(self, image, orig):
        targ_sz = orig.size
        image = image.resize(targ_sz, resample=PIL.Image.Resampling.BILINEAR)
        return image


class ColorizerFilter(BaseFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats, map_to_orig: bool = True):
        super().__init__(learn = learn, stats = stats)
        self.render_base = 16
        self.map_to_orig = map_to_orig

    def filter(self, orig_image, filtered_image, render_factor: int,post_process: bool = False):
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz)

        if self.map_to_orig:
            return self._post_process(model_image, orig_image, post_process )
        else:
            return self._post_process(model_image, filtered_image, post_process)

    def _transform(self, image):
        return image.convert('LA').convert('RGB')

    def add_intensity(self, img, intensity = 1.7):
        if intensity == 1:
            return img
        inter_const = 255. ** (1 - intensity)
        return (inter_const * (img ** intensity)).astype(np.uint8)

    def _post_process(self, raw_color, orig, post_process: bool):
        raw_color = self._unsquare(raw_color, orig)
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        if not post_process:
            color = self.add_intensity(color_np)
            blurred = cv2.GaussianBlur(orig_np, (5,5), 1)
            res_blur = cv2.addWeighted(orig_np, 0.75, blurred, 0.25, 0)
            color = cv2.addWeighted(res_blur, 0.5, color, 0.5, 0)
            combined = self.add_intensity(color, intensity=0.9)
            return PIL.Image.fromarray(combined)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)
        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)
        final = PIL.Image.fromarray(final)
        return final


class MasterFilter(BaseFilter):
    def __init__(self, filters, render_factor: int):
        self.filters = filters
        self.render_factor = render_factor

    def filter(self, orig_image, filtered_image, render_factor: int = None,post_process: bool = False):
        render_factor = self.render_factor if render_factor is None else render_factor
        for filter in self.filters:
            filtered_image = filter.filter(orig_image, filtered_image, render_factor, post_process)

        return filtered_image

torch.backends.cudnn.benchmark = True 

def get_transformed_image(binary_data, filtr, params):
    torch.cuda.empty_cache()
    orig_image = PIL.Image.open(io.BytesIO(binary_data)).convert("RGB")
    
    result = orig_image
    if params["clr_saturate_every_step"] == True:
        if params["compare"] == True:
            for _ in tqdm(range(params["steps"])):
                result = increase_color_saturation(image = filtr.filter(orig_image, result, render_factor = params["render_factor"], post_process = params["post_process"]), brite_k = params["clr_saturation_factor"], min_brite = params["line_color_limit"])
        else:
            for _ in tqdm(range(params["steps"])):
                result = increase_color_saturation(image = filtr.filter(result, result, render_factor = params["render_factor"], post_process = params["post_process"]), brite_k = params["clr_saturation_factor"], min_brite = params["line_color_limit"])
    else:
        if params["compare"] == True:
            for _ in tqdm(range(params["steps"])):
                result = filtr.filter(orig_image, result, render_factor = params["render_factor"], post_process = params["post_process"])
        else:
            for _ in tqdm(range(params["steps"])):
                result = filtr.filter(result, result, render_factor = params["render_factor"], post_process = params["post_process"])
        result = increase_color_saturation(image = result, brite_k = params["clr_saturation_factor"], min_brite = params["line_color_limit"])
    buf = io.BytesIO()
    result.save(buf, format = "PNG")
    b_data = buf.getvalue()
    result.close
    return b_data

def get_image_filtr(weights_name: str, artistic: bool = True, stats: tuple = imagenet_stats):
    if artistic:
        weights = weights_name
    else:
        weights = "ColorizeStable_gen"
    learn = gen_inference_deep(root_folder = Path('./'), weights_name = weights, stats=stats)
    filtr = MasterFilter([ColorizerFilter(learn=learn, stats=stats)], render_factor = 35)
    return filtr

def increase_color_saturation(image, brite_k = 0, min_brite = 0):
    pixels = image.load()
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            clr = pixels[x, y]
            if clr[0] + clr[1] + clr[2] > min_brite:
                pixels[x, y] = (max(0, 255 - (255 - clr[0]) * brite_k), max(0, 255 - (255 - clr[1]) * brite_k), max(0, 255 - (255 - clr[2]) * brite_k))
    return image

def Image_сolorizer(input_binary_image, params):
    filtr = get_image_filtr(weights_name = params["ckpt"], artistic = params["artistic"], stats = ([0.7137, 0.6628, 0.6519], [0.2970, 0.3017, 0.2979]))
    binary_image = get_transformed_image(input_binary_image, filtr = filtr, params = params)
    return binary_image