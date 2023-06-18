from Stable_diffusion import Stable_diffusion_text_to_image
from tqdm import tqdm
import os

orig_params = {
    "steps": 50,            #количество шагов выборки
    "sampler": "ddim",      #обработчик (доступно "plms", "dpm" и "ddim")
    "ddim_eta": 0.0,        #ddim η (от 0.0 до 1.0, η = 0.0 соответствует детерминированной выборке)
    "f": 8,                 #коэффициент понижающей дискретизации, чаще всего 8 или 16, если поставить 4, будет красиво, но учетверяться
    "scale": 9.0,           #безусловная навигационная величина: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    "ckpt": 0,              #выбор контрольной точки модели (0 или 1 для размерностей 512 или 768 соответственно)
    "seed": 42              #сид (для воспроизводимой генерации изображений)
}

params = orig_params

test_list = [
    #["steps", [10, 100, 50]],
    #["f", [4, 8]],
    #["scale", [4.0, 15.0, 9.0]],
    #["ckpt", [1, 0]],
    #["seed", [415, 758, 8613, 42]],
    #["sampler", ["plms", "dpm", "ddim"]],



    ["sampler", ["ddim"]],



    #["ddim_eta", [0.25, 0.75, 1.0, 0.0]],
]

caption = []
with open("C:\\Stable-Draw\\Neural_network_processing\\newdata.txt") as file_in:
    for line in file_in:
        caption.append(line.replace("\n", ""))

path1 = "C:\\Users\\Robolightning\\Desktop\\r3\\"

for tl in tqdm(test_list):
    for tp in tqdm(tl[1]):
        params = orig_params
        params[tl[0]] = tp
        path = path1 + tl[0] + "\\" + str(tp) + "\\"
        os.makedirs(path, exist_ok = True)
        for cap in tqdm(caption):
            w, h, binary_data = Stable_diffusion_text_to_image(cap, params) #описание и параметры генерации
            with open(path + cap + ".png", "wb") as f:
                f.write(binary_data)