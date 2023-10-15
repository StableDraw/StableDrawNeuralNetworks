from Neural_network_processing.Image_caption_generator import Gen_caption
from tqdm import tqdm
import os

orig_params = {
    "ckpt": "caption_huge_best.pt", #используемые чекпоинты (caption_huge_best.pt или caption_base_best.pt) #https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_base_best.pt
    "eval_cider": True,             #оценка с помощью баллов CIDEr
    "eval_bleu": False,             #оценка с помощью баллов BLEU
    "eval_args": "{}",              #аргументы генерации для оценки BLUE или CIDEr, например, "{"beam": 4, "lenpen": 0,6}", в виде строки JSON
    "eval_print_samples": False,    #печатать поколения образцов во время валидации
    "scst": False,                  #Обучение самокритичной последовательности
    "scst_args": "{}",              #аргументы генерации для обучения самокритичной последовательности в виде строки JSON
    "beam": 5,                      #балансировка (больше 0)
    "max_len_a": 0,                 #максимальная длина буфера a
    "max_len_b": 200,               #максимальная длина буфера b (больше 0)
    "min_len": 1,                   #минимальная длина буфера (при 46 могут быть проблемы, не знаю почему)
    "unnormalized": False,          #ненормализовывать
    "lenpen": 1,                    #(больше 0)
    "unkpen": 0,
    "temperature": 1.0,             #температура
    "match_source_len": False,      #сопоставлять с исходной длиной
    "no_repeat_ngram_size": 3,      #не повторять N-граммы размера
    "sampling_topk": 3,             #из скольки тоненов отбирать лучший (0 - не использовать сэмплирование)
    "seed": 7                       #инициализирующее значение для генерации
}

params = orig_params

test_list = [
    #["ckpt", ["caption_base_best.pt", "caption_huge_best.pt"]],
    #["eval_cider", [False, True]],
    #["eval_bleu", [True, False]],
    #["eval_print_samples", [True, False]],
    #["scst", [True, False]],
    #["beam", [10, 46, 2, 1, 5]],
    #["max_len_a", [10, 5, 2, 1, 46, 0]],
    #["max_len_b", [500, 1000, 10, 100, 1, 200]],
    #["min_len", [2, 0, 5, 10, 46, 1]],
    #["unnormalized", [True, False]],
    #["lenpen", [-5, -1, 5, 56, 0, 1]],
    #["unkpen", [-5, -1, 1, 5, 56, 0]],



    ["lenpen", [1]],
    ["unkpen", [56, 0]],


    
    ["temperature", [5.0, 0.0, 0.1, 30.0, 1.0]],
    ["match_source_len", [True, False]],
    ["no_repeat_ngram_size", [10, 0, 1, 40, 3]],
    ["sampling_topk", [10, 0, -3, 43, 3]],
    ["seed", [0, 736, 1, 36942, 7]]
]

img_containers = []
#код для считывания бинари даты из изображения

c_list = [
    ["face", "f_"],
    ["no face", "nf_"],
    ["not pro drawing", "npd_"],
    ["pro drawing", "pd"],
    ["pro line", "pl"],
    ["quick line", "ql"]
]

for cls in c_list:
    for root, dirs, files in os.walk("C:\\Stable-Draw\\test\\" + cls[0]):
        for f_name in files:
            with open(root + "\\" + f_name, "rb") as f:
                binary_data = f.read()
            img_containers.append([cls[1] + f_name, binary_data])

path1 = "C:\\Users\\Robolightning\\Desktop\\r4\\"

for tl in tqdm(test_list):
    for tp in tqdm(tl[1]):
        params = orig_params
        params[tl[0]] = tp
        path = path1 + tl[0] + "\\" + str(tp) + "\\"
        os.makedirs(path, exist_ok = True)
        for img_container in tqdm(img_containers):
            caption = Gen_caption(img_container[1], params)
            with open(path + img_container[0] + ".txt", "w", encoding='utf-8') as f:
                f.write(caption)
