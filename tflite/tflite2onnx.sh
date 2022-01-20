#!/bin/bash

printf "tflite2onnx convert start"

MODEL_NAME=$1
MODEL_TYPE=$2

pip install -U tf2onnx

python -m tf2onnx.convert \
        --opset 9 \
        --tflite ${MODEL_TYPE}/${MODEL_NAME}.tflite\
        --output ../onnx/${MODEL_TYPE}/${MODEL_NAME}_tflite.onnx \
        


printf "convert done"
