#!/bin/bash

printf "tf2onnx convert start"

MODEL_NAME=$1
MODEL_TYPE=$2

python -m tf2onnx.convert \
        --saved-model ./$MODEL_TYPE/${MODEL_NAME}_saved_model \
        --output ./$MODEL_TYPE/${MODEL_NAME}_tf.onnx \
        --opset 9


printf "convert done"
