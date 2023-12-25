import io
import os
import ssl
import cv2
import json
import time
import numpy
import base64
import asyncio
import requests
import websockets
import PIL
from PIL import Image
from dotenv import load_dotenv
from Image_caption_generator import Gen_caption
from Delete_background import Delete_background
from RealESRGAN import RealESRGAN_upscaler
from Stable_diffusion import Stable_diffusion_image_to_image as Stable_diffusion_2_0_image_to_image
from Stable_diffusionXL import Stable_diffusion_XL_image_to_image
from Stable_diffusion import Stable_diffusion_text_to_image as Stable_diffusion_2_0_text_to_image
from Stable_diffusionXL import Stable_diffusion_XL_text_to_image
from Stable_diffusion import Stable_diffusion_depth_to_image as Stable_diffusion_2_0_depth_to_image
from Stable_diffusion import Stable_diffusion_inpainting
from Stable_diffusion import Stable_diffusion_upscaler
from Stable_diffusion import Stable_diffusion_upscaler_xX
from Image_classifier import Get_image_class
from Image_сolorization import Image_сolorizer
from Kandinsky_2 import Kandinsky2_text_to_image
from Kandinsky_2 import Kandinsky2_image_to_image
from Kandinsky_2 import Kandinsky2_inpainting
from Kandinsky_2 import Kandinsky2_stylization
from Kandinsky_2 import Kandinsky2_mix_images
from Translate import Translator

chat_id = "-1001661093241"

with open("token.txt", "r") as f:
    TOKEN = f.read()

translator = Translator()

URL = "https://api.telegram.org/bot{}/".format(TOKEN)

task_list = [] #дескриптор сокета, тип задания, номер сообщения ТГ (id задания), user_id

user_id = "0" #Пока что мы не знаем id (убрать)

process = False
nprocess = False

user_settings = {
    "autotranslate": True,          #переводить автоматически
    "dest_lang": "ru",              #язык, на который переводить
    "autoclass": True,              #классифицировать автоматически
    "autofaceenchance": True,       #улучшать лица, если классификатор определил как фото лица
    "autoproclr": True,             #раскрашивать автоматически, если классификатор определил как профессиональный лайн
    "autoquickclr": True,           #раскрашивать автоматически, если классификатор определил как быстрый лайн
    "autobwcolorize": True,         #раскрашивать автоматически, если рисунок не цветной
    "autophotofacepreset": True,    #автоматически устанавливать пресет настроек для обработки фотографий, если классификатор определил как фото с лицом
    "autophotonofacepreset": True,  #автоматически устанавливать пресет настроек для обработки фотографий, если классификатор определил как фото без лица
    "autoproartpreset": True,       #автоматически устанавливать пресет настроек для обработки профессионального рисунка, если классификатор определил как профессиональный рисунок
    "autonoproartpreset": True,     #автоматически устанавливать пресет настроек для обработки непрофессионального рисунка, если классификатор определил как непрофессиональный рисунок
    "autoprolinepreset": True,      #автоматически устанавливать пресет настроек для обработки профессионального лайна, если классификатор определил как профессиональный лайн
    "autoquicklinepreset": True     #автоматически устанавливать пресет настроек для обработки быстрого лайна, если классификатор определил как быстрый лайн
    }

def send_document_to_tg(req_text, tgfile):
    req = requests.post(req_text, files = tgfile)
    content = req.content.decode("utf8")
    content_json = json.loads(content)
    message_id = str(content_json["result"]["message_id"])
    return message_id

def send_message_to_tg(req_text):
    req = requests.post(req_text)
    content = req.content.decode("utf8")
    content_json = json.loads(content)
    message_id = str(content_json["result"]["message_id"])
    return message_id

def del_prompt_about_drawing(prompt, rep_mess_id, noback):
    orig_p = prompt
    del_list = ['a sketch of an ',
                'a sketch of a ',
                'an outline of an ',
                'a outline of an ',
                'an outline of a ',
                'a outline of a ',
                'drawing of an',
                ' drawing of a',
                'a drawing of ',
                'an image of ',
                'a picture of ',
                "a continuous line of an",
                "a continuous line of a",
                "a continuous line of ",
                "a continuous line ",
                'a retro illustration of an ',
                'a retro illustration of a ',
                'an illustration of an ',
                'an illustration of a ',
                'an illustration of ',
                'a logo of ',
                'a sketch of ',
                'sketch of ',
                'drawing of ',
                'image of ',
                'picture of ',
                'illustration of ',
                'logo of',
                'logo ',
                'logo',
                ' icon',
                ' coloring page',
                ' outline style',
                ' illustration',
                ' isolated',
                'isolated']
    if noback == True:
        del_list.append(' on a white background')
        del_list.append(' with a white background')
    for dw in del_list:
        prompt = prompt.replace(dw, '')
    if prompt != orig_p:
        message_id = send_message_to_tg(URL + "sendMessage?text=" + orig_p + "&reply_to_message_id=" + rep_mess_id + "&chat_id=" + chat_id)
        time.sleep(0.3) #иметь ввиду, что тут слип, убрать его потом, после отключения от Телеги (убрать)
    else:
        message_id = rep_mess_id
    return prompt, message_id

def Prepare_img(path_dir, image_name, img_suf, no_gen, sim_suf, no_resize):
    frame_size = 10
    local_image = Image.open(path_dir + "\\" + image_name + "_" + str(img_suf) + ".png").convert("RGBA")
    b_image = local_image
    w, h = local_image.size
    rw, rh = w, h
    p_min = 2 * max(w, h) + min(w, h)
    clr_list = [[(255, 255, 255, 255), 0]]
    for x in range(w):
        clr = local_image.getpixel((x, 0))
        try:
            pos = next(i for i, (x, _) in enumerate(clr_list) if x == clr)
            clr_list[pos][1] += 1
        except:
            clr_list.append([clr, 0])
    for y in range(h):
        clr = local_image.getpixel((0, y))
        try:
            pos = next(i for i, (x, _) in enumerate(clr_list) if x == clr)
            clr_list[pos][1] += 1
        except:
            clr_list.append([clr, 0])
    for x in range(w):
        clr = local_image.getpixel((x, h - 1))
        try:
            pos = next(i for i, (x, _) in enumerate(clr_list) if x == clr)
            clr_list[pos][1] += 1
        except:
            clr_list.append([clr, 0])
    for y in range(h):
        clr = local_image.getpixel((w - 1, y))
        try:
            pos = next(i for i, (x, _) in enumerate(clr_list) if x == clr)
            clr_list[pos][1] += 1
        except:
            clr_list.append([clr, 0])
    need_crop = False
    for i in range(0, len(clr_list)):
        if clr_list[i][1] > p_min:
            need_crop = True
            clr = clr_list[i][0]
            break
    if need_crop:
        frame = True
        while frame:
            for x in range(w):
                new_clr = local_image.getpixel((x, 0))
                if new_clr != clr:
                    frame = False
                    break
            if frame:
                local_image = local_image.crop((0, 1, w, h)).convert("RGBA")
                h -= 1
                if h == 0:
                    buf = io.BytesIO()
                    b_image.save(buf, format = "PNG")
                    b_image.close()
                    binary_data = buf.getvalue()
                    return True, binary_data
        h_opt = rh - h
        frame = True
        while frame:
            for y in range(h):
                new_clr = local_image.getpixel((0, y))
                if new_clr != clr:
                    frame = False
                    break
            if frame:
                local_image = local_image.crop((1, 0, w, h)).convert("RGBA")
                w -= 1
        w_opt = rw - w
        frame = True
        while frame:
            for x in range(w):
                new_clr = local_image.getpixel((x, h - 1))
                if new_clr != clr:
                    frame = False
                    break
            if frame:
                local_image = local_image.crop((0, 0, w, h - 1)).convert("RGBA")
                h -= 1
        frame = True
        while frame:
            for y in range(h):
                new_clr = local_image.getpixel((w - 1, y))
                if new_clr != clr:
                    frame = False
                    break
            if frame:
                local_image = local_image.crop((0, 0, w - 1, h)).convert("RGBA")
                w -= 1
        dw, dh = rw / w, rh / h
        if no_resize:
            opt2_w, opt2_h = int(w * frame_size / 512), int(h * frame_size / 512)
            new_w, new_h = w + (opt2_w * 2), h + (opt2_h * 2)
            pd = 1
        else:
            new_w, new_h = 512, 512
            if w > h:
                pd = 512 / w
                h = int((512 - (frame_size * 2)) * h / w)
                w = 512 - (frame_size * 2)
                opt2_w = frame_size
                opt2_h = int((512 - h) / 2)
                is_w_bigger = True
            else:
                pd = 512 / h
                w = int((512 - (frame_size * 2)) * w / h)
                h = 512 - (frame_size * 2)
                opt2_h = frame_size
                opt2_w = int((512 - w) / 2)
                is_w_bigger = False
            local_image = local_image.resize((w, h), resample = Image.Resampling.LANCZOS).convert("RGBA")
        rimg = Image.new("RGBA", (new_w, new_h), clr)
        rimg.paste(local_image, (opt2_w, opt2_h),  local_image)
        if sim_suf == False:
            img_suf += 1
        if no_gen == True:
            rimg.save(path_dir + "\\" + "r_" + image_name + "_" + str(img_suf) + ".png")
        else:
            rimg.save(path_dir + "\\" + "c_" + image_name + "_" + str(img_suf) + ".png")
        new_w, new_h = int(w * dw), int(h * dh)
        w_opt, h_opt = int(w_opt * pd), int(h_opt * pd)
        if no_resize == False:
            if is_w_bigger == True:
                h_opt -= opt2_h
            else:
                w_opt -= opt2_w
    else:
        if need_crop == False:
            rimg = local_image
        else:
            local_image.close()
    buf = io.BytesIO()
    rimg.save(buf, format = "PNG")
    rimg.close()
    binary_data =  buf.getvalue()
    if need_crop == True:
        return [clr, new_w, new_h, w_opt, h_opt], binary_data
    return False, binary_data

def Restore_Image(bd, rbuf, path, restore_file_name):
    with open(path + "\\c_" + restore_file_name + "_restore.json", 'w') as f:
        f.write(json.dumps(rbuf))
    limg = Image.open(io.BytesIO(bd)).convert("RGBA")
    rimg = Image.new("RGBA", (rbuf[1], rbuf[2]), (rbuf[0][0], rbuf[0][1], rbuf[0][2], rbuf[0][3]))
    rimg.paste(limg, (rbuf[3], rbuf[4]), limg)
    buf = io.BytesIO()
    rimg.save(buf, format = "PNG")
    rimg.save(path + "\\r_" + restore_file_name + ".png")
    limg.close()
    rimg.close()
    return buf.getvalue()

def make_mask(img, path_to_save):
    img_array = numpy.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
    w, h = img.size
    A_img_array = numpy.zeros((h, w, 3), dtype = numpy.uint8)
    A_img_array[:, :, 0] = img_array[:, :, 3]
    A_img_array[:, :, 1] = img_array[:, :, 3]
    A_img_array[:, :, 2] = img_array[:, :, 3]
    cv2.imwrite(path_to_save, A_img_array)
    im_buf_arr = cv2.imencode(".png", A_img_array)[1]
    b_data = im_buf_arr.tobytes()
    return b_data

def colorize(init_img_binary_data, image_class):
    params = {
        "ckpt": "ColorizeArtistic_gen",    #Выбор модели ("ColorizeArtistic_gen", "ColorizeArtistic_gen_GrayScale", "ColorizeArtistic_gen_Sketch", "ColorizeArtistic_gen_Sketch2Gray")
        "steps": 1,                        #Количество шагов обработки (минимум 1)
        "compare": False,                  #Сравнивать с оригиналом
        "artistic": True,                  #Дополнительная модель для обработки
        "render_factor": 12,               #Фактор обработки (от 7 до 45) (лучше 12)
        "post_process": True,              #Постобработка
        "clr_saturation_factor": 5,        #Коэффициент увеличения цветовой насыщенности (1 - не добавлять насыщенность)
        "line_color_limit": 100,           #минимальная яркость пикселя, при которой цветовая насыщенность увеличиваться не будет (меньше для цифровых рисунков, больше для рисунков карандашом. 1 если лайн абсолютно чёрный)
        "clr_saturate_every_step": True    #Повышать цветовую насыщенность после каждого шага (играет роль только если количество шагов обработки больше 1)
        #max_dim не учитывается, поскольку в колоризатор встроен внутренний даунсемплер
    }

    if user_settings["autophotofacepreset"] == True and image_class == 0: #нужно настроить и убрать лишнее
        params["steps"] = 1
        params["ckpt"] = "ColorizeArtistic_gen_GrayScale"
        params["compare"] = False
        params["artistic"] = True
        params["render_factor"] = 12
        params["post_process"] = True
        params["clr_saturation_factor"] = 1
        params["line_color_limit"] = 100
        params["clr_saturate_every_step"] = True
    elif user_settings["autophotonofacepreset"] == True and image_class == 1:
        params["steps"] = 1
        params["ckpt"] = "ColorizeArtistic_gen_GrayScale"
        params["compare"] = False
        params["artistic"] = True
        params["render_factor"] = 12
        params["post_process"] = True
        params["clr_saturation_factor"] = 1
        params["line_color_limit"] = 100
        params["clr_saturate_every_step"] = True
    elif user_settings["autoproartpreset"] == True and image_class == 2:
        params["steps"] = 1
        params["ckpt"] = "ColorizeArtistic_gen"
        params["compare"] = False
        params["artistic"] = True
        params["render_factor"] = 12
        params["post_process"] = True
        params["clr_saturation_factor"] = 5
        params["line_color_limit"] = 100
        params["clr_saturate_every_step"] = True
    elif user_settings["autonoproartpreset"] == True and image_class == 3:
        params["steps"] = 1
        params["ckpt"] = "ColorizeArtistic_gen_GrayScale"
        params["compare"] = False
        params["artistic"] = True
        params["render_factor"] = 12
        params["post_process"] = True
        params["clr_saturation_factor"] = 1
        params["line_color_limit"] = 100
        params["clr_saturate_every_step"] = True
    elif user_settings["autoprolinepreset"] == True and image_class == 4:
        params["steps"] = 1
        params["ckpt"] = "ColorizeArtistic_gen_Sketch"
        params["compare"] = False
        params["artistic"] = True
        params["render_factor"] = 12
        params["post_process"] = True
        params["clr_saturation_factor"] = 2
        params["line_color_limit"] = 100
        params["clr_saturate_every_step"] = True
    elif user_settings["autoquicklinepreset"] == True and image_class == 5:
        params["steps"] = 1
        params["ckpt"] = "ColorizeArtistic_gen_Sketch"
        params["compare"] = False
        params["artistic"] = True
        params["render_factor"] = 12
        params["post_process"] = True
        params["clr_saturation_factor"] = 2
        params["line_color_limit"] = 40
        params["clr_saturate_every_step"] = True
    return Image_сolorizer(init_img_binary_data, params) #передаю путь к рабочей папке и имя файла

async def neural_processing(process, nprocess):
    if nprocess == True:
        return
    while process:
        nprocess = True
        if len(task_list) != 0:
            task = task_list.pop(0)
            websocket = task[0]
            task_type = task[1]
            img_suf = task[3]
            task_id = task[4]
            user_id = task[5]
            chain_id = task[6]
            final_file_name = task[7]
            init_img_binary_data = None
            postview = None
            print("Обработка началась")
            path_to_task_dir = "log\\" + user_id + "\\" + task_id
            if task_type != 't': #если в обработку передаётся изображение
                orig_img_name = task[2]
                img_name = orig_img_name + "_" + str(img_suf)
                if img_name[0] == 'c' and img_name[1] == '_':
                    need_restore = True
                    new_img_name = ""
                    if task_type == 'p' and task[10] == True:
                        if img_suf != 0:
                            new_img_suf = img_suf - 1
                        else:
                            new_img_suf = 0
                        rbufer, init_img_binary_data = Prepare_img(path_to_task_dir, orig_img_name[2:], new_img_suf, False, False, True)
                        img_name = orig_img_name + "_" + str(img_suf)
                        chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, {"document": ("c_" + img_name + ".png", init_img_binary_data)})
                    else:
                        with open(path_to_task_dir + "\\" + img_name + "_restore.json", 'r') as f:
                            rbufer = json.loads(f.read())

                        if img_suf == 0:
                            img_name = img_name[:-1] + "1"
                            img_suf == 1

                        fn = path_to_task_dir + "\\" + img_name + ".png"
                        if not(os.path.isfile(fn)) and img_name[:2] == "c_":
                            img_name = img_name[2:]
                        with open(fn, "rb") as f:
                            init_img_binary_data = f.read()
                elif os.path.exists(path_to_task_dir + "\\c_" + img_name + ".png"):
                    new_img_name = "c_"
                    need_restore = True
                    with open(path_to_task_dir + "\\c_" + img_name + "_restore.json", 'r') as f:
                        rbufer = json.loads(f.read())
                    with open(path_to_task_dir + "\\c_" + img_name + ".png", "rb") as f:
                        init_img_binary_data = f.read()
                    chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, { "document": ("c_" + img_name + ".png", init_img_binary_data) })
                elif task_type == 'p' or task_type == 'c':
                    if task_type == 'c':
                        rbufer, init_img_binary_data = Prepare_img(path_to_task_dir, orig_img_name, img_suf, False, False, False)
                        if rbufer != True or rbufer != False:
                            img_suf -= 1
                        img_name = orig_img_name + "_" + str(img_suf + 1)
                        chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, {"document": ("c_" + img_name + ".png", init_img_binary_data)})
                    else:
                        rbufer, init_img_binary_data = Prepare_img(path_to_task_dir, orig_img_name, img_suf, False, True, False)
                    if rbufer != True and rbufer != False:
                        new_img_name = "c_"
                        need_restore = True
                    else:
                        new_img_name = ""
                        need_restore = False
                    chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, {"document": (new_img_name + img_name + ".png", init_img_binary_data)})
                else:
                    with open(path_to_task_dir + "\\" + img_name + ".png", "rb") as f:
                        init_img_binary_data = f.read()
                    new_img_name = ""
                    need_restore = False
                img_name = new_img_name + img_name + ".png"
            img_suf += 1
            
            image_class = -1
            if init_img_binary_data != None and user_settings["autoclass"] == True:
                image_classes = Get_image_class(init_img_binary_data)
                image_class = image_classes[0]
                classes = [
                    "фото с лицом",
                    "фото без лица",
                    "профессиональный рисунок",
                    "непрофессиональный рисунок",
                    "профессиональный лайн",
                    "быстрый лайн"
                ]
                subclasses = [
                    "в цвете",
                    "чб"
                ]
                image_subclass = image_classes[1]
                chain_id = send_message_to_tg(URL + "sendMessage?text=" + "Определён класс: " + classes[image_class] + ", " + subclasses[image_subclass] + "&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id)
                if (not task_type in ['f', 'a', 'o']) and ((user_settings["autobwcolorize"] and image_subclass == 1) or (user_settings["autoproclr"] == True and image_class == 4) or (user_settings["autoquickclr"] == True and image_class == 5)):
                    postview = str(base64.b64encode(init_img_binary_data).decode("utf-8"))
                    init_img_binary_data = colorize(init_img_binary_data, image_class)
                    result_img = "colored_" + str(img_suf)
                    img_suf += 1
                    image = PIL.Image.open(io.BytesIO(init_img_binary_data)).convert("RGB")
                    w, h = image.size
                    image.save(path_to_task_dir + "\\" + result_img + ".png")
                    image.close()
                    chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, { "document": (result_img + ".png", init_img_binary_data) })

            if task_type == 'c': #если нужно сгенерировать описание
                noback = task[8]
                if rbufer == True: #если это просто одноцветный фон, то выдать описание "solid color background"
                    english_caption = "solid color background"
                else:
                    message_id = chain_id
                    params = {
                        "ckpt": "caption_huge_best.pt", #используемые чекпоинты (caption_huge_best.pt или caption_base_best.pt) #https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_base_best.pt
                        "eval_cider": True,             #оценка с помощью баллов CIDEr
                        "eval_bleu": False,             #оценка с помощью баллов BLEU
                        "eval_args": "{}",              #аргументы генерации для оценки BLUE или CIDEr, например, "{"beam": 4, "lenpen": 0,6}", в виде строки JSON
                        "eval_print_samples": False,    #печатать поколения образцов во время валидации
                        "scst": False,                  #Обучение самокритичной последовательности
                        "scst_args": "{}",              #аргументы генерации для обучения самокритичной последовательности в виде строки JSON
                        "beam": 5,                      #балансировка
                        "max_len_a": 0,                 #максимальная длина буфера a
                        "max_len_b": 100,               #максимальная длина буфера b
                        "min_len": 1,                   #минимальная длина буфера
                        "unnormalized": False,          #ненормализовывать
                        "lenpen": 1,
                        "unkpen": 0,
                        "temperature": 1.0,             #температура
                        "match_source_len": False,      #сопоставлять с исходной длиной
                        "no_repeat_ngram_size": 3,      #не повторять N-граммы размера
                        "sampling_topk": 3,             #из скольки тоненов отбирать лучший (0 - не использовать сэмплирование)
                        "seed": 7                       #инициализирующее значение для генерации
                    }#max_dim не обнаружено. Возможно даунсемплит где-то внутри
                    english_caption = Gen_caption(init_img_binary_data, params)
                    english_caption, chain_id = del_prompt_about_drawing(english_caption, message_id, noback)
                    with open(path_to_task_dir + "\\" + final_file_name + "_" + str(img_suf) + ".txt", "w") as f:
                        f.write(english_caption)
                chain_id = send_message_to_tg(URL + "sendMessage?text=" + english_caption + "&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id)
                if user_settings["autotranslate"] == True:                   
                    _, caption = translator.translate(english_caption, "en", user_settings["dest_lang"])
                    with open(path_to_task_dir + "\\" + final_file_name + "_ru_" + str(img_suf) + ".txt", "w") as f:
                        f.write(caption)
                    time.sleep(0.3) #иметь ввиду, что тут слип, убрать его потом, после отключения от Телеги (убрать)
                    chain_id = send_message_to_tg(URL + "sendMessage?text=" + caption + "&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id)
                else:
                    caption = english_caption
                if need_restore == True: #если нужно восстановление
                    with open(path_to_task_dir + "\\c_" + orig_img_name + "_" + str(img_suf) + "_restore.json", 'w') as f:
                        f.write(json.dumps(rbufer))
                resp_data = {
                    '0': 'c',
                    '1': task_id,
                    '2': caption,
                    '3': chain_id,
                    '4': new_img_name + orig_img_name,
                    '5': img_suf,
                    '6': english_caption
                }

            elif task_type == 'p': #если нужно сгенерировать изображение по изображению
                caption = task[8]
                Is_depth = task[9]
                Is_inpainting = task[10]
                Is_upscale = task[11]
                Is_upscale_xX = task[12]
                mask_binary_data = task[13]
                if postview == None:
                    postview = str(base64.b64encode(init_img_binary_data).decode('utf-8'))
                message_id = chain_id
                if Is_inpainting == True:
                    params = {
                        "version": "Kandinsky2.2",                  #"SD-2.0", "Kandinsky2.0", "Kandinsky2.1", "Kandinsky2.2"
                        "progress": True,                           #Только для Kandinsky < 2.2 и обработчика "p_sampler"
                        "dynamic_threshold_v": 99.5,                #Только для "Kandinsky2.0" и "dynamic_threshold"
                        "denoised_type": "dynamic_threshold",       #("dynamic_threshold", "clip_denoised") только для "Kandinsky2.0"
                        "negative_prior_prompt": "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature", #Только для Kandinsky > 2.0
                        "negative_prompt": "",                      #Только для Kandinsky > 2.0
                        "low_vram_mode": False,                     #Только для Kandinsky: режим для работы на малом количестве видеопамяти
                        "sampler": "ddim_sampler",                  #("ddim_sampler", "plms_sampler", "p_sampler") для всех моделей, кроме Kandinsky 2.2
                        "steps": 50,                                #Шаги от 0 до 50
                        "prior_steps": 25,                          #Только для Kandinsky > 2.0
                        "num_cols": 1,                              #Количество вариций изображений за одну генерацию
                        "ddim_eta": 0.0,                            #значения от 0.0 до 1.0, η = 0.0 соответствует детерминированной выборке, только для обработчика "ddim_sampler" и для всех версий, кроме Kandinsky 2.2
                        "guidance_scale": 10.0,                     #от 0.1 до 30.0
                        "prior_scale": 4,                           #Только для Kandinsky > 2.0
                        "strength": 0.9,                            #сила увеличения/уменьшения шума. 1.0 соответствует полному уничтожению информации в инициализирующем образе
                        "ckpt": "512-inpainting-ema.safetensors",   #выбор весов модели ("512-inpainting-ema.safetensors")
                        "seed": 42,                                 #от 0 до 1000000
                        "verbose": False,                           #понятия не имею что это
                        "max_dim": pow(2048, 2)                     #я не могу генерировать на своей видюхе картинки больше 2048 на 2048
                    }
                    if params["version"] == "SD-2.0":
                        binary_data = Stable_diffusion_inpainting(init_img_binary_data, mask_binary_data, caption, params) #передаю сокет, путь к рабочей папке, имя файла и параметры
                    else:
                        binary_data = Kandinsky2_inpainting(init_img_binary_data, mask_binary_data, caption, params)[0]
                        
                elif Is_upscale == True or Is_upscale_xX == True:
                    params = {
                        "model": "StableDiffusionx4Upscaler", #("StableDiffusionx4Upscaler", "StableDiffusionxLatentx2Upscaler")
                        "steps": 50,                            #Шаги DDIM, от 2 до 250
                        "ddim_eta": 0.0,                        #значения от 0.0 до 1.0, η = 0.0 соответствует детерминированной выборке
                        "guidance_scale": 9.0,                  #от 0.1 до 30.0
                        "ckpt": "x4-upscaler-ema.safetensors",  #выбор весов модели ("x4-upscaler-ema.safetensors", только для модели "StableDiffusionx4Upscaler")
                        "seed": 42,                             #от 0 до 1000000
                        "outscale": 4,                          #Величина того, во сколько раз увеличть разшрешение изображения (рекоммендуется 4 для "StableDiffusionx4Upscaler" и 2 для "StableDiffusionxLatentx2Upscaler")
                        "noise_augmentation": 20,               #от 0 до 350
                        "negative_prompt": None,                #отрицательное описание (если без него, то None)
                        "verbose": False,
                        "max_dim": pow(1024, 2)                 #я не могу генерировать на своей видюхе картинки больше 512 на 512 для x4 и 512 на 512 для x2
                    }
                    outscale = params["outscale"]
                    if need_restore:
                        rbufer[1] *= outscale
                        rbufer[2] *= outscale
                        rbufer[3] *= outscale
                        rbufer[4] *= outscale
                    if params["model"] == "StableDiffusionx4Upscaler":
                        binary_data = Stable_diffusion_upscaler(init_img_binary_data, caption, params)
                    elif params["model"] == "StableDiffusionx4Upscaler":
                        binary_data = Stable_diffusion_upscaler_xX(init_img_binary_data, caption, params)
                    else:
                        raise ValueError("Доступны только \"StableDiffusionx4Upscaler\" и \"StableDiffusionx4Upscaler\"")

                else:
                    params = {
                        "add_watermark": False, #Добавлять невидимую вотермарку
                        "version": "Kandinsky2.2", #Выбор версии: "SDXL-base-1.0", "SDXL-base-0.9" (недоступна для коммерческого использования), "SD-2.0", "SD-2.1", "SD-2.1-768", "SDXL-refiner-0.9" (недоступна для коммерческого использования, используется как модель 2 стадии, для первой непригодна),  "SDXL-refiner-1.0" (используется как модель 2 стадии, для первой непригодна), "Kandinsky2.0", "Kandinsky2.1", "Kandinsky2.2"
                        "ControlNET": True, #Только для "Kandinsky2.2"
                        "progress": True, #Только для Kandinsky < 2.2 и обработчика "p_sampler"
                        "Depth": False, #Использовать дополнительный слой глубины (только для версий "Kandinsky2.2" ControlNET и версий "SD-2.0")
                        "use_custom_ckpt": False, #Использовать свои веса для выбранной версии модели (для всех версий кроме SD-2.0)
                        "custom_ckpt_name": "512-depth-ema.safetensors", #Имя кастомной модели, если выбран "use_custom_ckpt". Является обязательным параметром для версии SD-2.0. (SD-2.0: "512-depth-ema.safetensors" для Depth == True, и "sd-v1-1.safetensors", "sd-v1-1-full-ema.safetensors", "sd-v1-2.safetensors", "sd-v1-2-full-ema.safetensors", "sd-v1-3.safetensors", "sd-v1-3-full-ema.safetensors", "sd-v1-4.safetensors", "sd-v1-4-full-ema.safetensors", "sd-v1-5.safetensors", "sd-v1-5-full-ema.safetensors" для Depth == False)
                        "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
                        "version2SDXL-refiner": False, #Только для версий SDXL-base: загрузить SDXL-refiner как модель для второй стадии обработки. Требует более длительной обработки и больше видеопамяти
                        "seed": 42, #Инициализирующее значение (может быть от 0 до 1000000000)
                        "negative_prompt": "", #Для всех моделей, кроме SDXL-base и Kandinsky 2.0: негативное описание
                        "negative_prior_prompt": "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature", #Только для Kandinsky > 2.0
                        "refiner": "SDXL-refiner-1.0", #Если "version2SDXL-refiner" выбран, то какую версию модели для второй стадии обработки загрузить: "SDXL-refiner-1.0", "SDXL-refiner-0.9"  (недоступна для коммерческого использования)
                        "refinement_strength": 0.15, #Сила вклада обработки на второй стадии (от 0.0 до 1.0)
                        "finish_denoising": True, #Завершить удаление шума рафинёром (только для моделей SDXL-base, если включён version2SDXL-refiner)
                        "h": 1024, #Высота желаемого изображения (от 64 до 2048, должна быть кратна 64, для "SD-2.0" рекомендуется 512)
                        "w": 1024, #Ширина желаемого изображения (от 64 до 2048, должна быть кратна 64, для "SD-2.0" рекомендуется 512)
                        "f": 8, #Коэффициент понижающей дискретизации, чаще всего 8 или 16 (можно 4, тогда есть риск учетверения, но красиво и жрёт больше видеопамяти) (От 4 до 64), если (4 можно, если w * h <= 1024 * 1024, иначе > 4)
                        "use_recommended_res": True, #Использовать рекомендованное для каждой модели разрешение генерации, вместо указанных выше
                        "sampler": "EulerEDMSampler", #Обработчик (("EulerEDMSampler", "HeunEDMSampler", "EulerAncestralSampler", "DPMPP2SAncestralSampler", "DPMPP2MSampler", "LinearMultistepSampler") только для моделей кроме Kandinsky и SD-2.0), (("ddim_sampler", "plms_sampler", "p_sampler") Только для Kandinsky < 2.2)
                        "ddim_eta": 0.0, #Только для моделей ("SD-2.0" и Kandinsky < 2.2) и обработчика ddim, η (η = 0.0 соответствует детерминированной выборке)
                        "dynamic_threshold_v": 99.5, #Только для "Kandinsky2.0" и "dynamic_threshold"
                        "denoised_type": "dynamic_threshold", #("dynamic_threshold", "clip_denoised") только для "Kandinsky2.0"
                        "model_type": "dpt_hybrid", #Только для модели "SD-2.0", тип модели ("dpt_large", "dpt_hybrid", "midas_v21", "midas_v21_small")
                        "verbose": True, #Не знаю что это
                        "s_churn": 0.0,  #Только для обработчиков "EulerEDMSampler" или "HeunEDMSampler" (от 0.0 до 1.0)
                        "s_tmin": 0.0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы меньше этого значения (от 0.0 до "sigma_max" и < "s_tmax")
                        "s_tmax": 999.0,  #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы больше этого значения (от "sigma_min" до "sigma_max" и > "s_tmin")
                        "s_noise": 1.0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler" или "EulerAncestralSampler" или "DPMPP2SAncestralSampler") и "s_churn" > 0 (от 0.0)
                        "eta": 1.0, #Только для обработчика "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
                        "order": 4, #Только для обработчика "LinearMultistepSampler" (от 1)
                        "force_i2i_resolution": False, #Если выбран, то размер итогового изображения для I2I генерации будет взят из параметров (w, h), а не у исходного изображения
                        "i2i_strength": 0.75, #вклад в генерацию модели в режиме I2I (от 0.0 до 1.0)
                        "i2i_prior_strength": 0.85, #Только для "Kandinsky2.2" с "ControlNET"
                        "i2i_negative_prior_strength": 1.0, #Только для "Kandinsky2.2" с "ControlNET"
                        "m_k": 8, #Коэффициент улучшения при постобработке (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
                        "aesthetic_score": 6.0, #Эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
                        "negative_aesthetic_score": 2.5, #Обратный эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
                        "custom_orig_size": False, #Если применён, то меняет размеры входного изображения на "orig_width" и "orig_heigt", иначе оставляет равними размерам желаемого изображения
                        "orig_width": 1024, #Ширина входного изображения, если установлен параметр "custom_orig_size" (от 16)
                        "orig_heigt": 1024, #Высота входного изображения, если установлен параметр "custom_orig_size" (от 16)
                        "crop_coords_top": 0, #Обрезка координат сверху (от 0)
                        "crop_coords_left": 0, #Обрезка координат слева (от 0)
                        "guider_discretization": "VanillaCFG", #Дискретизатор проводника? ("VanillaCFG", "IdentityGuider")
                        "sampling_discretization": "LegacyDDPMDiscretization", #Дискретизатор обработчика ("LegacyDDPMDiscretization", "EDMDiscretization")
                        "sigma_min": 0.03, #Только для "EDMDiscretization" дискритизатора обработчика
                        "sigma_max": 14.61, #Только для "EDMDiscretization" дискритизатора обработчика
                        "rho": 3.0, #Только для "EDMDiscretization" дискритизатора обработчика
                        "num_cols": 1, #Количество возвращаемых изображений (от 1 до 10, но, думаю, можно и больше при желании)
                        "guidance_scale": 9.0,
                        "prior_scale": 4, #Только для Kandinsky > 2.0
                        "steps": 40, #Количество шагов обработки (от 0 до 1000)
                        "prior_steps": 25, #Только для Kandinsky > 2.0
                        "max_dim": pow(2048, 2) # я не могу генерировать на своей видюхе картинки больше 2048 на 2048
                    }
                    if "Kandinsky" in params["version"]:
                        binary_data = Kandinsky2_image_to_image(init_img_binary_data, caption, params)[0]
                    elif params["version"] == "SD-2.0":
                        binary_data = Stable_diffusion_2_0_image_to_image(init_img_binary_data, caption, params)[0]
                    else:
                        if Is_depth == True:
                            params["Depth"] = True
                            params["custom_ckpt_name"] = "512-depth-ema.safetensors"
                            params["version"] = "SD-2.0"
                            params["w"] = 512
                            params["h"] = 512
                            binary_data = Stable_diffusion_2_0_depth_to_image(init_img_binary_data, caption, params)[0]
                        else:
                            binary_data = Stable_diffusion_XL_image_to_image(init_img_binary_data, caption, params)[0]
                result_img = final_file_name + "_" + str(img_suf)
                image = PIL.Image.open(io.BytesIO(binary_data)).convert("RGB")
                w, h = image.size
                image.save(path_to_task_dir + "\\" + result_img + ".png")
                image.close()
                if need_restore == True: #если нужно восстановление
                    chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, {"document": ("c_" + result_img + ".png", binary_data)})
                    binary_data = Restore_Image(binary_data, rbufer, path_to_task_dir, result_img)
                    result_img = "r_" + result_img
                    with open(path_to_task_dir + "\\" + result_img + ".png", "wb") as f:
                        f.write(binary_data)
                chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, { "document": (result_img + ".png", binary_data) })
                img = str(base64.b64encode(binary_data).decode('utf-8'))
                resp_data = {
                    '0': 'i',
                    '1': img,
                    '2': w,
                    '3': h,
                    '4': chain_id,
                    '5': new_img_name + final_file_name,
                    '6': task_id,
                    '7': postview,
                    '8': img_suf
                }

            elif task_type == 'f': #если нужно удалить фон у изображения
                if postview == None:
                    postview = str(base64.b64encode(init_img_binary_data).decode("utf-8"))
                params = {
                    "model": "DIS", #Доступно "U2NET" или "DIS"
                    "RescaleT": 320, #Только для модели U2NET
                    #Только для модели "DIS":
                    "ckpt": "isnet.pth",        # Выбор впретренированных весов модели ("isnet.pth", "isnet-general-use.pth")
                    "interm_sup": False,        # Указать, активировать ли контроль промежуточных функций
                    "model_digit": True,        # Выберите точность с плавающей запятой (устанавливает False или True точность числа с плавающей запятой)
                    "seed": 0,                  # Инициализирующее значение
                    "cache_size": [1024, 1024], # Кешированное входное пространственное разрешение, можно настроить на другой размер
                    "input_size": [1024, 1024], # Входной пространственный размер модели, обычно используют одно и то же значение params["cache_size"], что означает, что мы больше не изменяем размер изображений
                    "crop_size": [1024, 1024]   # Размер случайно обрезки из ввода, обычно он меньше, чем params["cache_size"], например, [920, 920] для увеличения данных
                #max_dim, возможно, даунсемплится по "*_size" параметрам выше. Не исследованно, и зависит от модели. По умолчанию, ограничений нет
                }
                binary_data = Delete_background(init_img_binary_data, params) #передаю путь к рабочей папке и имя файла
                result_img = final_file_name + "_" + str(img_suf)
                image = PIL.Image.open(io.BytesIO(binary_data)).convert("RGB")
                w, h = image.size
                image.save(path_to_task_dir + "\\" + result_img + ".png")
                image.close()
                if need_restore == True: #если нужно восстановление
                    result_path = path_to_task_dir + "\\c_" + result_img
                    chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, {"document": ("c_" + result_img + ".png", binary_data)})
                    rbufer[0] = (0, 0, 0, 0)
                    binary_data = Restore_Image(binary_data, rbufer, path_to_task_dir, result_img)
                    result_img = "r_" + result_img
                    with open(path_to_task_dir + "\\" + result_img + ".png", "wb") as f:
                        f.write(binary_data)
                chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, {"document": (result_img + ".png", binary_data)})
                img = str(base64.b64encode(binary_data).decode('utf-8'))
                resp_data = {
                    '0': 'i',
                    '1': img,
                    '2': w,
                    '3': h,
                    '4': chain_id,
                    '5': new_img_name + final_file_name,
                    '6': task_id,
                    '7': postview,
                    '8': img_suf
                }

            elif task_type == 'a': #если нужно апскейлить изображение
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
                    "gpu-id": None                      #Устройство gpu для использования (по умолчанию = None) может быть 0, 1, 2 для обработки на нескольких GPU
                    #на данный момент "max_dim": pow(1024, 2) ((для всех моделей, кроме "RealESRGAN_x2plus") и "outscale": 4), и pow(2048, 2) (для модели "RealESRGAN_x2plus" и "outscale": 2)
                }
                if user_settings["autofaceenchance"] == True and image_class == 0:
                    params["face_enhance"] = True
                outscale = params["outscale"]
                binary_data = RealESRGAN_upscaler(init_img_binary_data, params) #передаю путь к рабочей папке
                result_img = final_file_name + "_" + str(img_suf)
                image = PIL.Image.open(io.BytesIO(binary_data)).convert("RGB")
                w, h = image.size
                image.save(path_to_task_dir + "\\" + result_img + ".png")
                image.close()
                if need_restore == True: #если нужно восстановление
                    result_path = path_to_task_dir + "\\c_" + result_img
                    chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, {"document": ("c_" + result_img + ".png", binary_data)})
                    rbufer[1] *= outscale
                    rbufer[2] *= outscale
                    rbufer[3] *= outscale
                    rbufer[4] *= outscale
                    binary_data = Restore_Image(binary_data, rbufer, path_to_task_dir, result_img)
                    result_img = "r_" + result_img
                    with open(path_to_task_dir + "\\" + result_img + ".png", "wb") as f:
                        f.write(binary_data)
                chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, { "document": (result_img + ".png", binary_data) })
                img = str(base64.b64encode(binary_data).decode('utf-8'))
                resp_data = {
                    '0': 'i',
                    '1': img,
                    '2': w,
                    '3': h,
                    '4': chain_id,
                    '5': new_img_name + final_file_name,
                    '6': task_id,
                    '7': "",
                    '8': img_suf
                }

            elif task_type == 't': #если нужно сгенерировать изображение по описанию
                caption = task[2]
                params = {
                    "add_watermark": False, #Добавлять невидимую вотермарку
                    "version": "Kandinsky2.2", # Выбор модели: "SDXL-base-1.0", "SDXL-base-0.9", "Kandinsky2.0", "Kandinsky2.1", "Kandinsky2.2", (недоступна для коммерческого использования), "SD-2.0", "SD-2.1", "SD-2.1-768", "SDXL-refiner-0.9" (недоступна для коммерческого использования, используется как модель 2 стадии, для первой непригодна), "SDXL-refiner-1.0" (используется как модель 2 стадии, для первой непригодна)
                    "ControlNET": False, #Только для "Kandinsky2.2"
                    "use_flash_attention": False, #Только для "Kandinsky"
                    "progress": True, #Только для Kandinsky < 2.2 и обработчика "p_sampler"
                    "dynamic_threshold_v": 99.5, #Только для "Kandinsky2.0" и "dynamic_threshold"
                    "denoised_type": "dynamic_threshold", #("dynamic_threshold", "clip_denoised") только для "Kandinsky2.0"
                    "use_custom_ckpt": False, #Использовать свои веса для выбранной версии модели
                    "custom_ckpt_name": "v2-1_512-ema-pruned.safetensors", #Имя кастомной модели, либо (если выбран "use_custom_ckpt", обязательный параметр), либо (для модели "SD-2.0", как обязательный параметр. Может быть "v2-1_512-ema-pruned.safetensors", "v2-1_512-nonema-pruned.safetensors", "v2-1_768-ema-pruned.safetensors", "v2-1_768-nonema-pruned.safetensors")
                    "low_vram_mode": False, #Режим для работы на малом количестве видеопамяти
                    "version2SDXL-refiner": False, #Только для версий SDXL-base: загрузить SDXL-refiner как модель для второй стадии обработки. Требует более длительной обработки и больше видеопамяти
                    "seed": 42, #Инициализирующее значение (может быть от 0 до 1000000000)
                    "negative_prompt": "", #Для всех моделей, кроме (SDXL-base и Kandinsky 2.0): негативное описание
                    "negative_prior_prompt": "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature", #Только для Kandinsky > 2.0
                    "refiner": "SDXL-refiner-1.0", #Если "version2SDXL-refiner" выбран, то какую версию модели для второй стадии обработки загрузить: "SDXL-refiner-1.0", "SDXL-refiner-0.9"  (недоступна для коммерческого использования)
                    "refinement_strength": 0.15, #Сила вклада обработки на второй стадии (от 0.0 до 1.0)
                    "finish_denoising": True, #Завершить удаление шума рафинёром (только для моделей SDXL-base, если включён version2SDXL-refiner)
                    "h": 1024, #Высота желаемого изображения (от 64 до 2048, должна быть кратна 64) (512 для Kandinsky 2.0 и SD 512, 768 для Kandinsky 2.1 и SD 2.1 768 и 1024 для Kandinsky 2.2 и SDXL)
                    "w": 1024, #Ширина желаемого изображения (от 64 до 2048, должна быть кратна 64) (512 для Kandinsky 2.0 и SD 512, 768 для Kandinsky 2.1 и SD 2.1 768 и 1024 для Kandinsky 2.2 и SDXL)
                    "f": 8, #Коэффициент понижающей дискретизации, чаще всего 8 или 16 (можно 4, тогда есть риск учетверения, но красиво и жрёт больше видеопамяти) (От 4 до 64), если (4 можно, если w * h <= 1024 * 1024, иначе > 4)
                    "use_recommended_res": True, #Использовать рекомендованное для каждой модели разрешение генерации, вместо указанных выше
                    "sampler": "EulerEDMSampler", #Обработчик (Для SD > 2.0: "EulerEDMSampler", "HeunEDMSampler", "EulerAncestralSampler", "DPMPP2SAncestralSampler", "DPMPP2MSampler", "LinearMultistepSampler") и (Kandinsky < 2.2 или SD-2.0: "ddim_sampler", "plms_sampler", "p_sampler")
                    "ddim_eta": 0.05, #только для обработчика "ddim_sampler" и (SD-2.0 или Kandinsky < 2.2)
                    "s_churn": 0.0,  #Только для обработчиков "EulerEDMSampler" или "HeunEDMSampler" (от 0.0 до 1.0)
                    "s_tmin": 0.0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы меньше этого значения (от 0.0 до "sigma_max" и < "s_tmax")
                    "s_tmax": 999.0,  #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler") и "s_churn" > 0, обнуляет сигмы больше этого значения (от "sigma_min" до "sigma_max" и > "s_tmin")
                    "s_noise": 1.0, #Только для обработчиков ("EulerEDMSampler" или "HeunEDMSampler" или "EulerAncestralSampler" или "DPMPP2SAncestralSampler") и "s_churn" > 0 (от 0.0)
                    "eta": 1.0, #Только для обработчика "EulerAncestralSampler" или "DPMPP2SAncestralSampler" (от 0.0)
                    "order": 4, #Только для обработчика "LinearMultistepSampler" (от 1)
                    "m_k": 8, #Коэффициент улучшения при постобработке (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
                    "aesthetic_score": 6.0, #Эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
                    "negative_aesthetic_score": 2.5, #Обратный эстетический коэффициент (если активирован version2SDXL-refiner и модель SDXL-base) (понятия не имею от скольки до скольки он может быть, надо тестить)
                    "custom_orig_size": False, #Если применён, то меняет размеры входного изображения на "orig_width" и "orig_heigt", иначе оставляет равними размерам желаемого изображения
                    "orig_width": 1024, #Ширина входного изображения, если установлен параметр "custom_orig_size" (от 16)
                    "orig_heigt": 1024, #Высота входного изображения, если установлен параметр "custom_orig_size" (от 16)
                    "crop_coords_top": 0, #Обрезка координат сверху (от 0)
                    "crop_coords_left": 0, #Обрезка координат слева (от 0)
                    "guider_discretization": "VanillaCFG", #Дискретизатор проводника? ("VanillaCFG", "IdentityGuider")
                    "sampling_discretization": "LegacyDDPMDiscretization", #Дискретизатор обработчика ("LegacyDDPMDiscretization", "EDMDiscretization")
                    "sigma_min": 0.03, #Только для "EDMDiscretization" дискритизатора обработчика
                    "sigma_max": 14.61, #Только для "EDMDiscretization" дискритизатора обработчика
                    "rho": 3.0, #Только для "EDMDiscretization" дискритизатора обработчика
                    "num_cols": 1, #Количество возвращаемых изображений (от 1 до 10, но, думаю, можно и больше при желании)
                    "prior_scale": 4, #Только для Kandinsky > 2.0
                    "guidance_scale": 5.0, #Величина guidance (от 0.0 до 100.0)
                    "steps": 40, #Количество шагов обработки (от 0 до 1000)
                    "prior_steps": 25, #Только для Kandinsky > 2.0
                    #на данный момент "max_dim": 8 * pow(2048, 2) / f
                }
                if params["version"] == "SD-2.0":
                    binary_data = Stable_diffusion_2_0_text_to_image(caption, params)
                elif "Kandinsky" in params["version"]:
                    binary_data = Kandinsky2_text_to_image(caption, params)[0]
                else:
                    binary_data = Stable_diffusion_XL_text_to_image(caption, params)[0]
                image = PIL.Image.open(io.BytesIO(binary_data)).convert("RGB")
                w, h = image.size
                image.save(path_to_task_dir + "\\tpicture_1.png")
                image.close()
                img = str(base64.b64encode(binary_data).decode('utf-8'))
                chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, {'document': ('drawing_0.png', binary_data)})
                resp_data = {
                    '0': 'i',
                    '1': img,
                    '2': w,
                    '3': h,
                    '4': chain_id,
                    '5': final_file_name,
                    '6': task_id,
                    '7': "",
                    '8': "1"
                }

            if task_type == 'o': #если нужно покрасить изображение
                if postview == None:
                    postview = str(base64.b64encode(init_img_binary_data).decode("utf-8"))
                binary_data = colorize(init_img_binary_data, image_class)
                image = PIL.Image.open(io.BytesIO(binary_data)).convert("RGB")
                w, h = image.size
                image.close()
                result_img = final_file_name + "_" + str(img_suf)
                with open(path_to_task_dir + "\\" + result_img + ".png", "wb") as f:
                    f.write(binary_data)
                if need_restore == True: #если нужно восстановление
                    result_path = path_to_task_dir + "\\c_" + result_img
                    chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, {"document": ("c_" + result_img + ".png", binary_data)})
                    rbufer[0] = (0, 0, 0, 0)
                    binary_data = Restore_Image(binary_data, rbufer, path_to_task_dir, result_img)
                    result_img = "r_" + result_img
                    with open(path_to_task_dir + "\\" + result_img + ".png", "wb") as f:
                        f.write(binary_data)
                chain_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + chain_id + "&chat_id=" + chat_id, {"document": (result_img + ".png", binary_data)})
                img = str(base64.b64encode(binary_data).decode('utf-8'))
                resp_data = {
                    '0': 'i',
                    '1': img,
                    '2': w,
                    '3': h,
                    '4': chain_id,
                    '5': new_img_name + final_file_name,
                    '6': task_id,
                    '7': postview,
                    '8': img_suf
                }

            await websocket.send(json.dumps(resp_data))


        else:
            process = False
            nprocess = False

def build_connection(ws, u_id, istask):
    user_path = "log/" + u_id
    if os.path.isdir(user_path):
        if istask:
            file_time = round(os.path.getctime(user_path))
            now_tome = round(time.time())
            wait_time = now_tome - file_time
            with open(user_path + "/limit.txt", "r") as f:
                limit = int(f.read())
            if limit != -1:
                if wait_time > 86400: #в день
                    limit = 25 #дневная норма
                limit -= 1
                with open(user_path + "/limit.txt", "w") as f:
                    f.write(str(limit))
                if limit == -1:
                    resp_data = {
                        '0' : "t",
                        '1' : "У вас не осталось попыток на сегодня. Подождите " + str(86400 - wait_time) + " секунд, чтобы продолжить, либо оформите платную подписку на сервис"
                    }
                    ws.send(json.dumps(resp_data))
                    ws.close()
                    return False
    else:
        os.mkdir(user_path)
        with open(user_path + "/limit.txt", "w") as f:
            limit = 25
            if istask:
                limit = 24
            f.write(limit)
    return user_path

pre_process = False

async def pre_processing(websocket, dictData_list):
    global pre_process
    while pre_process == True:
        dictData = dictData_list.pop()
        if(dictData["type"] == "s"): #если настройки
            is_task = False
        else:
            is_task = True


            #обработка ошибок, временная
            if dictData["type"] != 't' and "chain_id" in dictData and dictData["chain_id"] == "" and "data" in dictData and (dictData["data"] in ("", {})) and (not("backgroung" in dictData) or (dictData["backgroung"] == "" or dictData["backgroung"] == {})):
                resp_data = {
                    '0' : "t",
                    '1' : "Извините, почему-то к нам ничего не пришло..."
                }
                await websocket.send(json.dumps(resp_data))
                pre_process == False
                break



        user_path = build_connection(websocket, user_id, is_task)
        if user_path == False:
            if task_pre_list == []:
                pre_process = False
            break

        if(dictData["type"] == "d"): #нужно описание
            if dictData["chain_id"] == "":
                binary_data = base64.b64decode(bytes(dictData["data"][22:], 'utf-8'))
                pillow_img = Image.open(io.BytesIO(binary_data)).convert("RGBA")
                (w, h) = pillow_img.size
                if dictData["backgroung"] == "" or dictData["backgroung"] == {}:
                    noback = True
                    background_img = Image.new('RGBA', (w, h), (255, 255, 255))
                else:
                    print("\nback")
                    noback = False
                    binary_data2 = base64.b64decode(bytes(dictData["backgroung"][22:], 'utf-8'))
                    background_img = Image.open(io.BytesIO(binary_data2)).convert("RGBA").resize((w, h))
                drawing_img = background_img
                drawing_img.paste(pillow_img, (0,0),  pillow_img)
                buf = io.BytesIO()
                drawing_img.save(buf, format = 'PNG')
                result_binary_data = buf.getvalue()
                message_id = send_document_to_tg(URL + "sendDocument?chat_id=" + chat_id, {"document": ("drawing_0.png", result_binary_data)})
                task_dir = user_path + "/" + message_id
                os.mkdir(task_dir)
                with open(task_dir + "/drawing_0.png", "wb") as f:
                    f.write(result_binary_data)

                with open(task_dir + "/drawing_info_0.txt", "w") as f: #сбор статистики по рисунку
                    if dictData["is_drawing"]:
                        f.write("1\n")
                    else:
                        f.write("0\n")
                    if dictData["sure"]:
                        f.write("1\n")
                    else:
                        f.write("0\n")
                    f.write(str(dictData["prims_count"]) + "\n")
                    f.write(str(dictData["dots_count"]))

                if noback == False:
                    with open(task_dir + "/background_0.png", "wb") as f:
                        f.write(binary_data2)
                    with open(task_dir + "/foreground_0.png", "wb") as f:
                        f.write(binary_data)
                task_id = message_id
                img_name = "drawing"
                img_suf = 0
            else:
                noback = False
                message_id = dictData["chain_id"]
                task_id = dictData["task_id"]
                img_name = dictData["img_name"]
                img_suf = int(dictData["img_suf"])
            task_list.append([websocket, 'c', img_name, img_suf, task_id, user_id, message_id, "AI_caption", noback]) #нужно описание
            
        elif dictData["type"] == "g": #нужна картина по описанию
            Is_depth = dictData["is_depth"]
            Is_upscale = dictData["is_upscale"]
            Is_upscale_xX = dictData["is_upscale_xX"]
            if Is_upscale or Is_upscale_xX == True:
                final_file_name = "big_image"
            else:
                final_file_name = "picture"
            if dictData["is_human_caption"] == False: #по машинному описанию
                Is_inpainting = False
                img_name = dictData["img_name"]
                img_suf = int(dictData["img_suf"])
                task_id = dictData["task_id"]
                message_id = dictData["chain_id"]
                task_dir = user_path + "\\" + task_id
                c_name = task_dir + "\\AI_caption_" + str(img_suf) + ".txt"
                if not(os.path.isfile(c_name)):
                    c_name = task_dir + "\\AI_caption_1.txt"
                with open(c_name, 'r') as f:
                    caption = f.read()
            else: #по человеческому описанию
                Is_inpainting = dictData["is_inpainting"]
                if dictData["chain_id"] == "":
                    binary_data = base64.b64decode(bytes(dictData["foreground"][22:], 'utf-8'))
                    pillow_img = Image.open(io.BytesIO(binary_data)).convert("RGBA")
                    (w, h) = pillow_img.size
                    if dictData["backgroung"] == "" or dictData["backgroung"] == {}:
                        noback = True
                        background_img = Image.new('RGBA', (w, h), (255, 255, 255))
                    else:
                        noback = False
                        binary_data2 = base64.b64decode(bytes(dictData["backgroung"][22:], 'utf-8'))
                        background_img = Image.open(io.BytesIO(binary_data2)).convert("RGBA").resize((w, h))
                    drawing_img = background_img
                    if Is_inpainting == True:
                        noback = True
                    else:
                        drawing_img.paste(pillow_img, (0, 0),  pillow_img)
                    buf = io.BytesIO()
                    drawing_img.save(buf, format = 'PNG')
                    result_binary_data = buf.getvalue()
                    task_id = send_document_to_tg(URL + "sendDocument?caption=" + dictData["prompt"] + "&chat_id=" + chat_id, {"document": ("drawing_0.png", result_binary_data)})
                    task_dir = user_path + "/" + task_id
                    os.mkdir(task_dir)
                    with open(task_dir + "/drawing_0.png", "wb") as f:
                        f.write(result_binary_data)

                    with open(task_dir + "/drawing_info_0.txt", "w") as f:#сбор статистики по рисунку
                        if dictData["is_drawing"]:
                            f.write("1\n")
                        else:
                            f.write("0\n")
                        if dictData["sure"]:
                            f.write("1\n")
                        else:
                            f.write("0\n")
                        f.write(str(dictData["prims_count"]) + "\n")
                        f.write(str(dictData["dots_count"]))

                    if noback == False:
                        with open(task_dir + "/background_0.png", "wb") as f:
                            f.write(binary_data2)
                        with open(task_dir + "/foreground_0.png", "wb") as f:
                            f.write(binary_data)
                    img_name = "drawing"
                    img_suf = 0
                    need_make_text_file = True
                else:
                    task_id = dictData["task_id"]
                    task_dir = user_path + "/" + task_id
                    if not os.path.exists(task_dir + "/Human_caption_0.txt"):
                        need_make_text_file = True
                        message_id = send_message_to_tg(URL + "sendMessage?text=" + dictData["prompt"] + "&reply_to_message_id=" + dictData["chain_id"] + "&chat_id=" + chat_id)
                        time.sleep(0.3) #иметь ввиду, что тут слип, убрать его потом, после отключения от Телеги (убрать)
                    else:
                        need_make_text_file = False
                        message_id = dictData["chain_id"]
                    img_name = dictData["img_name"]
                    img_suf = int(dictData["img_suf"])
                lang, result_text = translator.translate(dictData["prompt"])
                if lang != "en":
                    with open(task_dir + "\\Human_caption_ru_" + str(img_suf) + ".txt", "w") as f:
                        f.write(dictData["prompt"])
                    time.sleep(0.3)
                    message_id = send_message_to_tg(URL + "sendMessage?text=" + result_text + "&reply_to_message_id=" + task_id + "&chat_id=" + chat_id)
                else:
                    message_id = task_id
                if need_make_text_file:
                    with open(task_dir + "\\Human_caption_" + str(img_suf) + ".txt", "w") as f:
                        f.write(result_text)
                caption = result_text
            if Is_inpainting == True:
                mask = make_mask(pillow_img, task_dir + "\\mask_" + str(img_suf) + ".png")
                message_id = send_document_to_tg(URL + "sendDocument?&reply_to_message_id=" + message_id + "caption=C маской&chat_id=" + chat_id, {"document": ("mask_" + str(img_suf) + ".png", mask)})
            else:
                mask = ""
            task_list.append([websocket, "p", img_name, img_suf, task_id, user_id, message_id, final_file_name, caption, Is_depth, Is_inpainting, Is_upscale, Is_upscale_xX, mask]) #дескриптор сокета, тип задания, номер сообщения ТГ (id задания), user_id, номер последнего ответа ТГ
            
        elif(dictData["type"] == "b"): #нужно удалить фон у изображения
            if dictData["chain_id"] == "":
                binary_data = base64.b64decode(bytes(dictData["data"][22:], 'utf-8'))
                pillow_img = Image.open(io.BytesIO(binary_data)).convert("RGBA")
                buf = io.BytesIO()
                pillow_img.save(buf, format = 'PNG')
                result_binary_data = buf.getvalue()
                message_id = send_document_to_tg(URL + "sendDocument?caption=Изображение для удаления фона&chat_id=" + chat_id, {'document': ('drawing_0.png', result_binary_data)})
                task_id = message_id
                task_dir = user_path + "/" + message_id
                os.mkdir(task_dir)

                with open(task_dir + "/picture_0.png", "wb") as f:
                    f.write(result_binary_data)
                img_name = "picture"
                img_suf = 0
            else:
                message_id = dictData["chain_id"]
                task_id = dictData["task_id"]
                img_name = dictData["img_name"]
                img_suf = int(dictData["img_suf"])
            task_list.append([websocket, "f", img_name, img_suf, task_id, user_id, message_id, "object"]) #дескриптор сокета, тип задания, номер сообщения ТГ (id задания), user_id, номер последнего ответа ТГ, ширину и высоту исходного изображения

        elif(dictData["type"] == "a"): #нужно апскейлить изображение
            if dictData["chain_id"] == "":
                binary_data = base64.b64decode(bytes(dictData["data"][22:], 'utf-8'))
                pillow_img = Image.open(io.BytesIO(binary_data)).convert("RGBA")
                buf = io.BytesIO()
                pillow_img.save(buf, format = 'PNG')
                result_binary_data = buf.getvalue()
                message_id = send_document_to_tg(URL + "sendDocument?caption=Изображение для апскейлинга&chat_id=" + chat_id, { "document": ('drawing_0.png', result_binary_data)})
                task_id = message_id
                task_dir = user_path + "/" + message_id
                os.mkdir(task_dir)
                with open(task_dir + "/picture_0.png", "wb") as f:
                    f.write(result_binary_data)
                img_name = "picture"
                img_suf = 0
            else:
                message_id = dictData["chain_id"]
                task_id = dictData["task_id"]
                img_name = dictData["img_name"]
                img_suf = int(dictData["img_suf"])
            task_list.append([websocket, "a", img_name, img_suf, task_id, user_id, message_id, "big_image"]) #дескриптор сокета, тип задания, номер сообщения ТГ (id задания), user_id, номер последнего ответа ТГ, ширину и высоту исходного изображения
            
        elif(dictData["type"] == "t"): #нужно сгенерировать изображение по описанию
            prompt = dictData["prompt"]
            task_id = send_message_to_tg(URL + "sendMessage?text=" + prompt + "&chat_id=" + chat_id)
            lang, result_text = translator.translate(prompt)
            if lang != "en":
                time.sleep(0.3) #иметь ввиду, что тут слип, убрать его потом, после отключения от Телеги (убрать)
                message_id = send_message_to_tg(URL + "sendMessage?text=" + result_text + "&reply_to_message_id=" + task_id + "&chat_id=" + chat_id)
            else:
                message_id = task_id
            task_dir = user_path + "\\" + task_id
            os.mkdir(task_dir)
            with open(task_dir + "\\Human_caption_0.txt", "w") as f:
                f.write(result_text)
            img_suf = 0
            task_list.append([websocket, "t", result_text, img_suf, task_id, user_id, message_id, "tpicture"]) #дескриптор сокета, тип задания, текст описания, номер сообщения ТГ (id задания), user_id, номер последнего ответа ТГ, Использовать ли SD2, или Dall-e 2
        
        elif(dictData["type"] == "c"): #нужно покрасить изображение
            if dictData["chain_id"] == "":
                binary_data = base64.b64decode(bytes(dictData["data"][22:], 'utf-8'))
                pillow_img = Image.open(io.BytesIO(binary_data)).convert("RGBA")
                buf = io.BytesIO()
                pillow_img.save(buf, format = 'PNG')
                result_binary_data = buf.getvalue()
                message_id = send_document_to_tg(URL + "sendDocument?caption=Изображение для окрашивания&chat_id=" + chat_id, {'document': ('drawing_0.png', result_binary_data)})
                task_id = message_id
                task_dir = user_path + "/" + message_id
                os.mkdir(task_dir)

                with open(task_dir + "/picture_0.png", "wb") as f:
                    f.write(result_binary_data)
                img_name = "picture"
                img_suf = 0
            else:
                message_id = dictData["chain_id"]
                task_id = dictData["task_id"]
                img_name = dictData["img_name"]
                img_suf = int(dictData["img_suf"])
            task_list.append([websocket, "o", img_name, img_suf, task_id, user_id, message_id, "colored"]) #дескриптор сокета, тип задания, номер сообщения ТГ (id задания), user_id, номер последнего ответа ТГ, ширину и высоту исходного изображения

        tls = len(task_list)
        '''
        if tls > 1:
            client_message = "Ожидайте. Человек перед вами: " + str(tls - 1)
        else:
            client_message = "Обработка начнётся прямо сейчас..."
        resp_data = {
            '0' : "t",
            '1' : client_message
        }
        await websocket.send(json.dumps(resp_data))
        '''
        process = True
        await neural_processing(process, nprocess)
        if task_pre_list == []:
            pre_process = False

task_pre_list = []

async def handler(websocket): #здесь нужно формировать список текущих заданий, полученных от пользователей
    try:
        while True:
            try:
                jsonData = await websocket.recv()
            except:
                try:
                    task_list.pop(next(i for i, (x, _) in enumerate(task_list) if x == websocket))
                except:
                    print("Клиент слишком рано разорвал соединение")
                break
            dictData = json.loads(jsonData)
            task_pre_list.append(dictData)
            global pre_process
            if pre_process == False:
                pre_process = True
                await pre_processing(websocket, task_pre_list) #вызывать её, если задание получено, передавать в неё список dictData, пока он не пуст, пусть его обрабатывает
            else:
                pre_process = True
    finally:
        print("Соединение разорвано со стороны пользователя")
        task_list.pop(next(i for i, (x, _) in enumerate(task_list) if x == websocket))

pre_process = False
if __name__ == "__main__":
    #translators.preaccelerate()
    load_dotenv()
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain("domain_name.crt", "private.key")
    start_server = websockets.serve(handler, "stabledraw.com", 8081, ssl = ssl_context, max_size = None)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()