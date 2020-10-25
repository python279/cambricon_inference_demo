#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
import torchvision.models as models
import cv2
from PIL import Image
import logging
from AbsDeepLearningInference import AbsDeepLearningInference

USE_MLU = os.environ.get("USE_MLU", False)
if USE_MLU:
    import torch_mlu
    import torch_mlu.core.mlu_quantize as mlu_quantize
    import torch_mlu.core.mlu_model as ct


class TorchResnet50InferenceBase(AbsDeepLearningInference):
    device = None
    model = None

    def __init__(self,  model_file="./models/resnet50.pth", test_img="./test.jpg"):
        self.device = self.get_device()
        self.model_file = model_file
        self.test_img = test_img
        self.quantification()
        self.warm_up()

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def get_device(self):
        return torch.device('cpu')

    def quantification(self):
        pass

    def warm_up(self):
        # 加载模型并测试
        with torch.no_grad():
            self.model = models.resnet50()
            self.model.load_state_dict(torch.load(self.model_file), strict=False)
            self.model.eval().float()
            self.model.to(self.device)
        self.test()

    def test(self):
        # 推理测试图片
        request = {'data':  self.test_img}
        logging.info("warm_up: inference result={}".format(self.inference(request)))

    def inference(self, request):
        imgfile = request.get("data")
        img = cv2.imread(imgfile)
        batch_size = 1

        with torch.no_grad():
            # 推理数据预处理
            sized = cv2.resize(img, (224, 224))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            sized = ((sized / 255) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            sized = np.transpose(sized, (2, 0, 1))
            sized = sized.astype(np.float32)
            input_img = torch.from_numpy(np.stack([sized] * batch_size))

            # 模型推理
            input_img = input_img.to(self.device)
            out = self.model(input_img)
            out2 = self.softmax(out.cpu().numpy())
            return out2.argmax()


class TorchResnet50InferenceOnCPU(TorchResnet50InferenceBase):
    def __init__(self):
        super(TorchResnet50InferenceOnCPU, self).__init__()


class TorchResnet50InferenceOnGPU(TorchResnet50InferenceBase):
    def __init__(self):
        super(TorchResnet50InferenceOnGPU, self).__init__()

    def get_device(self):
        return torch.device('cuda')

    # GPU是否需要量化？
    #def quantification(self):
    #    pass


class TorchResnet50InferenceOnMLU270(TorchResnet50InferenceBase):
    def __init__(self):
        super(TorchResnet50InferenceOnMLU270, self).__init__()

    def get_device(self):
        # MLU270
        ct.set_core_version("MLU270")
        # MLU270有16个core，如果使用多线程，这里的core number可以设置成16/threads
        ct.set_core_number(16)
        return ct.mlu_device()

    def quantification(self):
        pass


def best_resnet50_inference():
    if USE_MLU:
        return TorchResnet50InferenceOnMLU270()
    elif torch.cuda.device_count():
        return TorchResnet50InferenceOnGPU()
    else:
        return TorchResnet50InferenceOnCPU()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    best_resnet50_inference()
