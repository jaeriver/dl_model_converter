import os
import numpy as np
import onnxruntime as rt
import tensorflow as tf
import tf2onnx
import boto3
from transformers import BertTokenizer, TFBertForQuestionAnswering
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bucket', type=str)
parser.add_argument('--model', default='bert', type=str)
args = parser.parse_args()
model_name = args.model
bucket_name = args.bucket

def get_model(model_name, bucket_name):
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    
    prefix = 'tf/' + model_name + '/'
    for object in bucket.objects.filter(Prefix = prefix):
        print(prefix)
        if object.key == prefix:
            os.makedirs(os.path.dirname(object.key), exist_ok=True)
            continue;
        bucket.download_file(object.key, object.key)
    
    return model_name
 
model_path = get_model(model_name, bucket_name)


tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir=None, local_files_only=True)
model = TFBertModel.from_pretrained(model_path, cache_dir=None, local_files_only=True)

input_spec = (
    tf.TensorSpec((None,  None), tf.int32, name="input_ids"),
    tf.TensorSpec((None,  None), tf.int32, name="token_type_ids"),
    tf.TensorSpec((None,  None), tf.int32, name="attention_mask")
)

_, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec, opset=13, output_path="bert.onnx")
