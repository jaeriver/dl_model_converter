import torchvision.models as models
import torch.onnx

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
parser.add_argument('--model_type',default='image_classification' , type=str)

args = parser.parse_args()

model_name = args.model
model_type = args.model_type

models_detail = {
    'resnet50':models.resnet50(pretrained=True),
    'inception_v3':models.inception_v3(pretrained=True),
    'mobilenet_v2':models.mobilenet_v2(pretrained=True),
    'efficientnetb0':models.efficientnet_b0(pretrained=True),
    'vgg16' : models.vgg16(pretrained=True)
}

import os
folder_path = f"./{model_type}"
try:
    os.mkdir(folder_path)
except:
    pass

model = models_detail[model_name]

saved_model_dir = f'./{model_type}/{model_name}.pth'
torch.save(model, saved_model_dir)
print(saved_model_dir," : compelete saved")
