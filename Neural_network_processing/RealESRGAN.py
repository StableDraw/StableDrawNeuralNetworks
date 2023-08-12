import cv2
import io
import os
import numpy as np
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def Upscale(binary_data, args):
    # определяет модели в соответствии с выбранной моделью
    model_name = args["model"]
    if model_name == "RealESRGAN_x4plus":  # модель x4 RRDBNet
        model = RRDBNet(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_block = 23, num_grow_ch = 32, scale = 4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == "RealESRNet_x4plus":  # модель x4 RRDBNet
        model = RRDBNet(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_block = 23, num_grow_ch = 32, scale = 4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == "RealESRGAN_x4plus_anime_6B":  # модель x4 RRDBNet с 6 блоками
        model = RRDBNet(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_block = 6, num_grow_ch = 32, scale = 4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == "RealESRGAN_x2plus":  # модель x2 RRDBNet
        model = RRDBNet(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_block = 23, num_grow_ch=32, scale = 2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == "realesr-animevideov3":  # модель x4 VGG-стиля (размера XS)
        model = SRVGGNetCompact(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_conv = 16, upscale = 4, act_type = 'prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == "realesr-general-x4v3":  # модель x4 VGG-стиля (размера S)
        model = SRVGGNetCompact(num_in_ch = 3, num_out_ch = 3, num_feat = 64, num_conv = 32, upscale = 4, act_type = "prelu")
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    model_path = os.path.join('weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # путь к модели будет обновлён
            model_path = load_file_from_url(url = url, model_dir = os.path.join(ROOT_DIR, 'weights'), progress = True, file_name = None)
    # использовать dni для контроля силы удаления шума
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and args["denoise_strength"] != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args["denoise_strength"], 1 - args["denoise_strength"]]
    # восстановитель
    upsampler = RealESRGANer(
        scale = netscale,
        model_path = model_path,
        dni_weight = dni_weight,
        model = model,
        tile = args["tile"],
        tile_pad = args["tile_pad"],
        pre_pad = args["pre_pad"],
        half = not args["fp32"],
        gpu_id = args["gpu-id"])
    if args["face_enhance"]:  # Использовать GFPGAN для улучшения лиц
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale = args["outscale"],
            arch = "clean",
            channel_multiplier = 2,
            bg_upsampler = upsampler)
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(binary_data))), cv2.COLOR_RGB2BGR)
    try:
        if args["face_enhance"]:
            _, _, output = face_enhancer.enhance(img, has_aligned = False, only_center_face = False, paste_back = True)
        else:
            output, _ = upsampler.enhance(img, outscale = args["outscale"])
    except RuntimeError as error:
        print("Ошибка", error, "\nЕсли у вас появляется ошибка \"CUDA out of memory\", попробуйте постаивть параметру \"tile\" меньшее значение")
    else:
        im_buf_arr = cv2.imencode(".png", output)[1]
        result_binary_data = im_buf_arr.tobytes()
    torch.cuda.empty_cache()
    return result_binary_data