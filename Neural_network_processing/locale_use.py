from RealESRGAN import RealESRGAN_upscaler
import os
import io
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    
    it = 10
    start = 0

    tp = "C:\\repos\\Real-ESRGAN\\experiments\\debug_train_RealESRGANx2plus_400k_B12G4_pairdata\\visualization\\"
    for j in tqdm(range(start, it + 1, 5)):
        params = {
            "model": "RealESRGAN_x2plus",       #Модель для обработки ("RealESRGAN_x4plus" - модель x4 RRDBNet, "RealESRNet_x4plus" - модель x4 RRDBNet, "RealESRGAN_x4plus_anime_6B" - модель x4 RRDBNet с 6 блоками, "RealESRGAN_x2plus" - модель x2 RRDBNet, "realesr-animevideov3" - модель x4 VGG-стиля (размера XS), "realesr-general-x4v3" - модель x4 VGG-стиля (размера S)) 
            "denoise_strength": 0.5,            #Сила удаления шума. 0 для слабого удаления шума (шум сохраняется), 1 для сильного удаления шума. Используется только для модели "realesr-general-x4v3"
            "outscale": 2,                      #Величина того, во сколько раз увеличть разшрешение изображения (модель "RealESRGAN_x2plus" x2, остальные x4)
            "tile": 0,                          #Размер плитки, 0 для отсутствия плитки во время тестирования
            "tile_pad": 10,                     #Заполнение плитки
            "pre_pad": 0,                       #Предварительный размер заполнения на каждой границе
            "face_enhance": False,              #Использовать GFPGAN улучшения лиц
            "fp32": True,                       #Использовать точность fp32 во время вывода. По умолчанию fp16 (половинная точность)
            "alpha_upsampler": "realesrgan",    #Апсемплер для альфа-каналов. Варианты: realesrgan | bicubic
            "gpu-id": None,                      #Устройство gpu для использования (по умолчанию = None) может быть 0, 1, 2 для обработки на нескольких GPU
            #на данный момент "max_dim": pow(1024, 2) ((для всех моделей, кроме "RealESRGAN_x2plus") и "outscale": 4), и pow(2048, 2) (для модели "RealESRGAN_x2plus" и "outscale": 2)
            "temp_param": str(j)
        }

        dp = tp + str(j) + "k"
        if not os.path.exists(dp):
            os.mkdir(dp)
        for root, _, files in os.walk(tp + "test\\"):  
            for filename in files:
                with open(tp + "test\\" + filename, "rb") as f:
                    init_img_binary_data = f.read()
                binary_data = RealESRGAN_upscaler(init_img_binary_data, params)
                Image.open(io.BytesIO(init_img_binary_data)).convert("RGBA").resize((4096, 4096), 4).save(dp + "\\" + filename[:-4] + " small.png")
                Image.open(io.BytesIO(binary_data)).convert("RGBA").resize((4096, 4096), 4).save(dp + "\\" + filename[:-4] + " big.png")