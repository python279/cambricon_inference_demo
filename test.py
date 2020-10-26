#!/usr/bin/env python
# coding: utf-8

import requests
import base64
import time
from concurrent.futures import ThreadPoolExecutor


def predict(i):
    start = time.time()
    with open("./test.jpg", "rb") as f:
        file_base64 = base64.b64encode(f.read())
        r = requests.post("http://localhost:5005/api/v1.0/predict",
                      json={'data': {'images':[{'content': file_base64.decode('ascii')}]}})
        print(r.json())
    print(time.time() - start)


with ThreadPoolExecutor(max_workers=4) as tpe:
    tpe.map(predict, range(10))
