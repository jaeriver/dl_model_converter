import os
import numpy as np
import onnxruntime as rt
import tf2onnx
from transformers import BertTokenizer, TFBertModel
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',default=1 , type=int)
args = parser.parse_args()
batch_size = args.batch_size

tf_model_path = 'tf/bert'    

tokenizer = BertTokenizer.from_pretrained(tf_model_path, cache_dir=None, local_files_only=True)
model = TFBertModel.from_pretrained(tf_model_path, cache_dir=None, local_files_only=True)

# switch the input_dict to numpy
sentence = "This is Fake Dataset for testing NLP Tokenizing"
test_batch = [sentence for i in range(batch_size)]
encoded_input = tokenizer(test_batch, 
                          return_tensors='tf',
                          padding=True,
                          truncation=True,
                          max_length=512)
input_dict_np = {k: v.numpy() for k, v in encoded_input.items()}

tf_results = model(encoded_input)
output_names = list(tf_results.keys())


model_name = "bert.onnx"
opt = rt.SessionOptions()
sess = rt.InferenceSession(model_name)
start_time = time.time()
onnx_results = sess.run(output_names, input_dict_np)
print(time.time() - start_time)
print(onnx_results)
