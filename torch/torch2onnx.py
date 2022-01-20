import torch.onnx

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
parser.add_argument('--model_type',default='image_classification' , type=str)
parser.add_argument('--batchsize',default=1 , type=int)

args = parser.parse_args()

model_name = args.model
model_type = args.model_type
batch_size = args.batchsize

model = torch.load(f'./{model_type}/{model_name}.pth')

# ------------------------ export -----------------------------
output_onnx = f'../onnx/{model_type}/{model_name}_torch.onnx'
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["input0"]
output_names = ["output0"]
inputs = torch.randn(batch_size, 3, 224, 224)


torch_out = torch.onnx._export(model, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names=output_names)

print("torch to onnx convert done")
