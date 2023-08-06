from __future__ import print_function, division
import torch
import PIL
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import U2NET
import numpy as np
from skimage import transform, color
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from io import BytesIO
from DIS_models.data_loader_cache import normalize, im_preprocess 
from DIS_models import *

#MODEL_PATH_URL = "https://drive.google.com/uc?id=1KyMpRjewZdyYfxHPYcd-ZbanIXtin0Sn"
#gdown.download(MODEL_PATH_URL, "saved_models/isnet.pth", use_cookies=False)

def load_image(image_bytes, params, transform):
    orig = Image.open(BytesIO(image_bytes)).convert("RGB")
    im = np.array(orig)
    im, im_shp = im_preprocess(im, params["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return orig, transform(im).unsqueeze(0), shape.unsqueeze(0) # Создание наборов изображений и формы

def build_model(params, device, restore_model):
    net = ISNetDIS()
    if(params["model_digit"] == "half"): # конвертация в половинную точность
        net.half()
        for layer in net.modules():
            if isinstance(layer, BatchNorm2d):
                layer.float()
    net.to(device)
    net.load_state_dict(torch.load("saved_models/" + restore_model, map_location = device))
    net.to(device)
    net.eval()  
    return net

def predict(net,  inputs_val, shapes_val, params, device): #Получено изображение Image, вычисление маски
    net.eval()
    if(params["model_digit"] == "full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)
    inputs_val_v = Variable(inputs_val, requires_grad = False).to(device) # Обернуть входные данные в переменную
    ds_val = net(inputs_val_v)[0] # список из 6 результатов
    pred_val = ds_val[0][0, :, :, :] # B x 1 x H x W    # Нам нужен первый, который является наиболее точным прогнозом
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode = 'bilinear')) # восстановить пространственный размер предсказания до исходного размера изображения
    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi) # Максимум = 1
    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8) # Получаем маску, которая необходима

class RescaleT(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,sample):
        imidx, image, label = sample['imidx'], sample['image'],sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size,int):
            if h > w:
                new_h, new_w = self.output_size*h/w,self.output_size
            else:
                new_h, new_w = self.output_size,self.output_size*w/h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        # img = transform.resize(image,(new_h,new_w),mode='constant')
        # lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

        img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
        lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

        return {'imidx':imidx, 'image':img,'label':lbl}

class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,flag=0):
        self.flag = flag

    def __call__(self, sample):

        imidx, image, label =sample['imidx'], sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        if(np.max(label)<1e-6):
            label = label
        else:
            label = label/np.max(label)

        # change the color space
        if self.flag == 2: # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0],image.shape[1],6))
            tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
            if image.shape[2]==1:
                tmpImgt[:,:,0] = image[:,:,0]
                tmpImgt[:,:,1] = image[:,:,0]
                tmpImgt[:,:,2] = image[:,:,0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
            tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
            tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
            tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
            tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
            tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
            tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
            tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
            tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

        elif self.flag == 1: #with Lab color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))

            if image.shape[2]==1:
                tmpImg[:,:,0] = image[:,:,0]
                tmpImg[:,:,1] = image[:,:,0]
                tmpImg[:,:,2] = image[:,:,0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

            tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
            tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
            tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

        else: # with rgb color
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
            image = image/np.max(image)
            if image.shape[2]==1:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
            else:
                tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
                tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
                tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

        tmpLbl[:,:,0] = label[:,:,0]


        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}

class SalObjDataset(Dataset):
    def __init__(self, binary_data, transform=None):
        self.binary_data = binary_data
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = np.array(PIL.Image.open(BytesIO(self.binary_data)).convert("RGB"))
        imidx = np.array([0])
        label = np.zeros(np.zeros(image.shape).shape[0:2])[:,:,np.newaxis]
        sample = {'imidx':imidx, 'image':image, 'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample

#Директория для весов
model_dir = "saved_models\\u2net.pth"

def U2NET_Delete_background(binary_data, params):
    test_salobj_dataset = SalObjDataset(binary_data = binary_data, transform = transforms.Compose([RescaleT(params["RescaleT"]), ToTensorLab(flag = 0)])) #Загрузчик даных
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size = 1, shuffle = False, num_workers = 1)
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location = "cuda"))
    net.eval()
    for data in test_salobj_dataloader: #вывод для каждого изображения
        inputs = data["image"]
        inputs = inputs.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        pred = net(inputs)[0][:, 0, :, :] #нормализация
        mi = torch.min(pred)
        pred = ((pred - mi) / (torch.max(pred) - mi)).squeeze()
        predict_np = pred.cpu().data.numpy()
        im = Image.fromarray(predict_np * 255).convert('L')
        orig = Image.open(BytesIO(binary_data)).convert("RGB")
        image = np.asarray(orig)
        imo = im.resize((image.shape[1], image.shape[0]), resample = Image.BILINEAR)
        orig.putalpha(imo)
        buf = BytesIO()
        orig.save(buf, format = "PNG")
        result_binary_data = buf.getvalue()
        torch.cuda.empty_cache()
    return result_binary_data

def DIS_Delete_background(binary_data, params):
    checkpoint_list = [
        "isnet.pth",
        "isnet-general-use.pth"
    ]
    restore_model = checkpoint_list[params["ckpt"]]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class GOSNormalize(object): #Нормализация Image при помощи torch.transforms
        def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
            self.mean = mean
            self.std = std

        def __call__(self,image):
            image = normalize(image, self.mean, self.std)
            return image
    transform = transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])
    net = build_model(params, device, restore_model)
    orig, image_tensor, orig_size = load_image(binary_data, params, transform) 
    mask = predict(net, image_tensor, orig_size, params, device)
    image = np.asarray(orig)
    im = Image.fromarray(mask).convert('L')
    imo = im.resize((image.shape[1], image.shape[0]), resample = Image.Resampling.BILINEAR)
    orig.putalpha(imo)
    buf = BytesIO()
    orig.save(buf, format = "PNG")
    result_binary_data = buf.getvalue()
    torch.cuda.empty_cache()
    return result_binary_data

def Delete_background(binary_data, params):
    if params["model"] == "U2NET":
        result_binary_data = U2NET_Delete_background(binary_data, params)
    else:
        result_binary_data = DIS_Delete_background(binary_data, params)
    return result_binary_data