from __future__ import print_function, division
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models, transforms

def Get_image_class(binary_data):
    cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 6)
    model = model_ft.to(device)
    checkpoint = torch.load("models\\classifier\\classifier.pth", map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    data_transforms = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model.eval()
    with torch.no_grad():
        sample = data_transforms(Image.open(io.BytesIO(binary_data)).convert("RGB"))
        inputs = sample.to(device)
        outputs = model(inputs.unsqueeze(0))
        _, preds = torch.max(outputs, 1)
        cl = int(preds[0])
    return(cl)