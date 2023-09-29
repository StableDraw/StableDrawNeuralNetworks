from typing import List
from typing import Optional


def colorizer(init_img_binary_data: bytes, params: dict) -> bytes:
    return init_img_binary_data


def delete_background(init_img_binary_data: bytes, params: dict) -> bytes:
    return init_img_binary_data


def upscaler(init_img_binary_data: bytes, caption: Optional[str], params: dict) -> bytes:
    return init_img_binary_data


def image_to_image(init_img_binary_data: bytes, caption: str, params: dict) -> List[bytes]:
    return List[init_img_binary_data]

# Генерация по тексту (Text to image) (4 разные нейронки, у них разные параметры, в комментариях всё указано, требуется максимум внимания)
# Принимает описание и параметры. Возвращает список изображений


def text_to_image(caption: str, params: dict) -> List[bytes]:
    return

# Генерация описания для изображения (Image captioning) (1 нейронка, параметры только для неё. Но скоро будет заменена другой нейронкой)
# Принимает изображение и параметры. Возвращает строку описания


def image_captioning(init_img_binary_data: bytes, caption: str, params: dict) -> str:
    return caption


def image_classification(init_img_binary_data: bytes) -> List[int]:
    return


def translation(input_text: str, source_lang: str = "", dest_lang: str = "en") -> tuple:
    return (source_lang, input_text)

# Перерисовка области изображения (Inpainting) (2 модели, нужно внимательно читать комментарии к параметра, чтобы понять что к чему относится)
# Принимает на вход исходное изображение, изображение маски (где содержимое маски может иметь любой цвет, но значение альфаканала равно 255, а область вне маски является прозрачное, то есть альфаканал в этих местах равен 0), строку описания и словарь параметров. Возвращает список изображений


def inpainting(init_img_binary_data: bytes, mask_binary_data: bytes, caption: str, params: dict) -> List[bytes]:
    return

# Стилизация (Style transfer) (1 модель пока что, скоро будет вторая)
# Принимает на вход изображение контента, изображение стиля, строку описания и словарь параметров. Возвращает список изображений


def stylization(content_binary_data: bytes, style_binary_data: bytes, prompt: str, params: dict) -> List[bytes]:
    return

# Совмещение изображений (Image fusion) (1 модель, но у неё 2 версии, внимательно читайте комментарии к параметрам, они могу относиться к разным версиям)
# Принимает первое и второе изображения для совмещения, описание первого и второго изображения для совмещения и словарь параметров. Возвращает список изображений


def image_fusion(img1_binary_data: bytes, img2_binary_data: bytes, prompt1: str, prompt2: str, params: dict) -> List[bytes]:
    return
