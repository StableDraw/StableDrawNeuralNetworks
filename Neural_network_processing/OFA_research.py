from Image_caption_generator import Gen_caption
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

    "score_reference": False,
    "print_alignment": False,
    "sampling": False,
    "sampling_topk": 3, #из скольки тоненов отбирать лучший (больше 0)
    "sampling_topp": -1.0, #(от -1.0 до 1.0, при 0.5 проблемы)
    "diverse_beam_groups": -1, #(<= 0)
    "diverse_beam_strength": 0.5,
    "match_source_len": False,
    "diversity_rate": -1, #(<= 0)
    "full_context_alignment": False,
    "constraints": False,

    "beam": 5,                      #балансировка (больше 0)
    "max_len_a": 0,                 #максимальная длина буфера a
    "max_len_b": 200,               #максимальная длина буфера b (больше 0)
    "min_len": 1,                   #минимальная длина буфера
    "unnormalized": False,          #ненормализовывать
    "lenpen": 1,                    #(больше 0)
    "unkpen": 0,
    "temperature": 1.0,             #температура (больше 0.0)
    "diverse_beam_groups": -1,
    "no_repeat_ngram_size": 3,      #не повторять N-граммы размера
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
    #["lenpen", [-5, -1, 5, 56, 1]],
    #["unkpen", [-5, -1, 1, 5, 56, 0]],
    #["temperature", [5.0, 0.1, 30.0, 1.0]],
    #["sampling", [True, False]],
    #["sampling_topk", [10, 43, 3]],
    #["sampling_topp", [0.0, -0.4, -0.2, -0.5, -0.1, 0.1, 0.5, -1.0]],
    #["diverse_beam_groups", [0, -3, -43, -1]],
    #["diverse_beam_strength", [-1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 10.0, 43.0]],
    #["diversity_rate", [0, -3, -10, -43, -1]],

    
    ["diversity_rate", [-10, -43, -1]],
    ["full_context_alignment", [True, False]],
    ["constraints", [True, False]],
    ["no_repeat_ngram_size", [-1, -10, -40, -3, 10, 0, 1, 40, 3]],
    ["seed", [-1, -43, -36942, 0, 736, 1, 36942, 7]]
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
