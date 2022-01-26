from tvm import relay
import onnx
import tvm 

import time
import argparse
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
parser.add_argument('--model_type',default='image_classification' , type=str)
parser.add_argument('--framework',default='tf' , type=str)
parser.add_argument('--batchsize',default=1 , type=int)
parser.add_argument('--imgsize',default=224 , type=int)
parser.add_argument('--arch',default='arm' , type=str)
parser.add_argument('--export',default=False , type=bool)

args = parser.parse_args()

model_name = args.model
model_type = args.model_type
batch_size = args.batchsize
size = args.imgsize
arch_type = args.arch
export = args.export
framework = args.framework

def make_dataset(batch_size,size):
    # Have to fix due to famework 
    # image_shape = (size, size,3)
    image_shape = (3,size, size)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

if arch_type == "llvm":
    target = "llvm"
else:
    target = tvm.target.arm_cpu()

ctx = tvm.cpu()
onnx_model = onnx.load(f'./convert_onnx/{model_type}/{model_name}_{framework}.onnx')

data,image_shape = make_dataset(batch_size,size)

# Have to fix due to famework 
shape_dict = {"input0": data.shape}

##### Convert tensorflow model 
print("ONNX model imported to relay frontend.")
mod, params = relay.frontend.from_onnx(onnx_model, shape=shape_dict)


##### TVM compile 
print("-"*10,"Compile style : Graph runtime ","-"*10)
build_time = time.time()
with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build_module.build(mod, target=target, params=params)

print(type(graph), type(mod), type(params))

if export :
    #lib.export_library(f"./{model_name}_{batch_size}_{arch_type}.tar")
    import os
    try:
        try:
            target_path = f"./{arch_type}"
            os.mkdir(target_path)
        except:
            pass
        folder_path = f"./{arch_type}/{model_name}_{batch_size}_{framework}"
        os.mkdir(folder_path)
    except:
        print("Already Exist")
        pass

    name = f"./{arch_type}/{model_name}_{batch_size}_{framework}/model"
    graph_fn, lib_fn, params_fn = [name+ext for ext in ('.json','.tar','.params')]
    lib.export_library(lib_fn)
    with open(graph_fn, 'w') as f:
        f.write(graph)
    with open(params_fn, 'wb') as f:
        f.write(relay.save_param_dict(params))
    
    print("Compile using TVM")