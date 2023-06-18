import io
from fairseq import metrics
from fairseq.tasks import register_task
import torch
import numpy
from fairseq import utils, tasks, checkpoint_utils
from tasks.mm_tasks.caption import CaptionTask
from PIL import Image
from torchvision import transforms
from utils.eval_utils import eval_caption
from pytorch_lightning import seed_everything

# Подготовка теста
def encode_text(text, task, length = None, append_bos = False, append_eos = False):
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    s = task.tgt_dict.encode_line(line = task.bpe.encode(text), add_if_not_exist = False, append_eos = False).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Формирование выхода для задания описания
def construct_sample(image: Image, task, patch_resize_transform):
    pad_idx = task.src_dict.pad()
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", task, append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": numpy.array(['42']),
        "net_input": 
        {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample

# Функция, меняющая FP32 на FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype = torch.half)
    return t

def Gen_caption(binary_data, params):
    overrides = {
        "bpe_dir": "./utils/BPE",               #путь до BPE
        "eval_cider_cached_tokens": "corpus",   #путь к кэшированному файлу cPickle, используемому для расчета оценок CIDEr
        "sampling": True,                       #использовать ли семплирование
        "clip_model_path": "../../checkpoints/clip/ViT-B-16.pt",
        "vqgan_model_path": "../../checkpoints/vqgan/last.ckpt",
        "vqgan_config_path": "../../checkpoints/vqgan/model.yaml"
    }
    if params["sampling_topk"] == 0:
        overrides["sampling"] == False
    tasks.register_task('caption', CaptionTask)
    use_cuda = torch.cuda.is_available()
    use_fp16 = False
    seed_everything(params["seed"])
    # Загрузка претренированных ckpt и config
    print("Загрузка претренированных чекпоинтов и настроек конфигурации...")
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(utils.split_paths(params["ckpt"]), arg_overrides = params | overrides)
    # Перемещение моделей на GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)
    # Инициализация генератора
    generator = task.build_generator(models, cfg.generation)
    # Преобразование изображений
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    patch_resize_transform = transforms.Compose(
        [
            lambda image: image.convert("RGB"),
            transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation = Image.Resampling.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]
    )
    image = Image.open(io.BytesIO(binary_data))
    # Построение выходного образца и подготовка GPU, если доступна CUDA
    print("Построение выходного образца...")
    sample = construct_sample(image, task, patch_resize_transform)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    # Запуск эволюционного шага для описания
    print("Вычисление описания...")
    with torch.no_grad():
        caption = eval_caption(task, generator, models, sample)[0][0]['caption']
        torch.cuda.empty_cache()
        return caption