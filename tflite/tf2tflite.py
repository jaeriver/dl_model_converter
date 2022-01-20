# how to run : example
# python3 tf2tflite.py --model mobilenet --model_type classification

import tensorflow as tf

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model',default='resnet50' , type=str)
parser.add_argument('--model_type',default='image_classification' , type=str)

args = parser.parse_args()
model_name = args.model
model_type = args.model_type

saved_model_dir = f"../tf/{model_type}/{model_name}_saved_model"

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

# Save the model.
with open(f'./{model_type}/{model_name}.tflite', 'wb') as f:
  f.write(tflite_model)
print("TF to TFLITE convert done")
