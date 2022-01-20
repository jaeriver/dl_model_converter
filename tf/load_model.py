import os
import shutil

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.applications import nasnet,mobilenet,mobilenet_v2,resnet,inception_v3,vgg16

models_detail = {
    # 'nasnetmobile':nasnet.NASNetMobile(weights='imagenet'),
    'mobilenet':mobilenet.MobileNet(weights='imagenet'),
    # 'mobilenet_v2':mobilenet_v2.MobileNetV2(weights='imagenet'),
    # 'resnet50':resnet.ResNet50(weights='imagenet'),
    # 'inception_v3':inception_v3.InceptionV3(weights='imagenet',include_top=False),
    # 'vgg16':vgg16.VGG16(weights='imagenet')
}

import argparse

results = None
parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
args = parser.parse_args()
model_name = args.model

saved_model_dir = f'./image_classification/{model_name}_saved_model'

def load_save_model(model_name,saved_model_dir):
    model = models_detail[model_name]
    shutil.rmtree(saved_model_dir, ignore_errors=True)
    try:
        model.save(saved_model_dir, include_optimizer=False, save_format='tf')
        print(saved_model_dir," : complete saved ")
    except:
        print("NOT saved")

load_save_model(model_name,saved_model_dir)
