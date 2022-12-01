#!/usr/bin/env python
# coding: utf-8
from io import BytesIO
from urllib import request

from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)

    x = np.array(img, dtype=np.float32)
    x *= 1./255
    X = np.array([x])
    return X

url = 'https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg'

def predict(url):
    img = download_image(url)
    X = prepare_image(img, (150, 150))
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    return preds[0].tolist()


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

if __name__ == '__main__':
    print(predict(url))
