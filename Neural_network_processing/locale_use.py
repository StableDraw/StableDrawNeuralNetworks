from Stable_diffusion import Stable_diffusion_inpainting
import os
import time

if __name__ == '__main__':
    params = {
        "ddim_steps": 50,           #Шаги DDIM, от 0 до 50
        "ddim_eta": 0.0,            #значения от 0.0 до 1.0, η = 0.0 соответствует детерминированной выборке
        "scale": 10.0,              #от 0.1 до 30.0
        "strength": 0.9,            #сила увеличения/уменьшения шума. 1.0 соответствует полному уничтожению информации в инициализирующем образе
        "ckpt": 0,                  #выбор весов модели (0)
        "seed": 42,                 #от 0 до 1000000
        "verbose": False,
        "max_dim": pow(4096, 2)  # я не могу генерировать на своей видюхе картинки больше 512 на 512
    }

    start = time.time()
    path = "C:\\Users\\Robolightning\\Desktop\\Учёба\\Восьмой семестр\\Курсовой проект 3\\Изображения\\2048\\"
    filename = "anime.png"
    if ".png" in filename:
        print(filename)
        caption = filename[:-4]
        with open(path + filename, "rb") as f:
            init_img_binary_data = f.read()
        with open(path + "mask.png", "rb") as f:
            mask_binary_data = f.read()
        binary_data = Stable_diffusion_inpainting(init_img_binary_data, mask_binary_data, caption, params)
        with open("C:\\Users\\Robolightning\\Desktop\\lam\\3\\" + filename, "wb") as f:
            f.write(binary_data)
    end = time.time() - start ## собственно время работы программы
    print("End\n" + str(end))
    with open("C:\\Users\\Robolightning\\Desktop\\lam\\3\\" + "time.txt", "w") as f:
        f.write(str(end)) ## вывод времени