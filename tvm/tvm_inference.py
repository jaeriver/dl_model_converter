from tvm import relay
import numpy as np 
import tvm 
from tvm.contrib import graph_executor
from tvm.contrib import graph_runtime

import tvm.testing 
import time
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
parser.add_argument('--model_type',default='image_classification' , type=str)
parser.add_argument('--framework',default='tf' , type=str)
parser.add_argument('--batchsize',default=1 , type=int)
parser.add_argument('--imgsize',default=224 , type=int)
parser.add_argument('--arch',default='arm' , type=str)

args = parser.parse_args()

model_name = args.model
model_type = args.model_type
batch_size = args.batchsize
size = args.imgsize
arch_type = args.arch
framework = args.framework

def make_dataset(batch_size,size):
    image_shape = (size, size,3)
    # image_shape = (3,size, size)
    data_shape = (batch_size,) + image_shape

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

    return data,image_shape

if arch_type == "llvm":
    target = "llvm"
else:
    target = tvm.target.arm_cpu()

ctx = tvm.cpu()
data,image_shape = make_dataset(batch_size,size)

# Have to fix due to framework 
shape_dict = {"input0": data.shape}

load_model = time.time()
graph_fn = f"./{arch_type}/{model_name}_{batch_size}_{framework}/model.json"
lib_fn = f"./{arch_type}/{model_name}_{batch_size}_{framework}/model.tar"
params_fn = f"./{arch_type}/{model_name}_{batch_size}_{framework}/model.params"

loaded_graph = open(graph_fn).read()
loaded_mod = tvm.runtime.load_module(lib_fn)
loaded_params = open(params_fn, "rb").read()

loaded_rt = tvm.contrib.graph_runtime.create(loaded_graph, loaded_mod, ctx)
loaded_rt.load_params(loaded_params)
print('load_model time', (((time.time() - load_model) ) * 1000),"ms")


measurements = 5
iter_times = []
for i in range(measurements):
    start_time = time.time()
    loaded_rt.run(data = data)
    print(f"TVM {model_name}-{batch_size} inference_time : ",(time.time()-start_time)*1000,"ms")
    iter_times.append(time.time() - start_time)

print("="*10,"time.time Module","="*10)
print(f"TVM model {model_name}-{batch_size}-{framework} inference latency : mean",np.mean(iter_times) * 1000 ,"ms", "& media : ",np.median(iter_times) * 1000,"ms" )
print("\n")



data_tvm = tvm.nd.array(data.astype('float32'))

e = loaded_rt.module.time_evaluator("run", ctx, number=5, repeat=1)
t = e(data_tvm).results
t = np.array(t) * 1000
   
print("="*10,"TVM - time evaluator module","="*10)
print('{} - {} (batch={}): mean {} ms '.format(model_name, framework, batch_size, t.mean()))
