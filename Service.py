#!/usr/bin/env python
# coding: utf-8

import os
from PIL import Image
import cv2
import logging
import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import TorchResnet50Inference


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
best_resnet50_inference = TorchResnet50Inference.best_resnet50_inference()
app = Flask(__name__)


def flask_return_result(code, msg, result=None):
    return jsonify({'code': code, 'msg': msg, 'data': result})


@app.route('/health')
def health():
    return flask_return_result('00000', 'OK')


@app.route('/api/v1.0/predict', methods=['POST'])
def predict():
    def _predict():
        try:
            j = request.get_json()
            file_base64 = j['data']['images'][0]['content']
            file_str = base64.b64decode(file_base64)
            image = BytesIO(file_str)
            r = best_resnet50_inference.inference(request={'data': image})
            return flask_return_result('00000', 'OK', str(r))
        except Exception as e:
            return flask_return_result('40001', e.message)
    return _predict()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)
