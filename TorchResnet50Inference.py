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
    USE_FUSION = True


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

    @staticmethod
    def image_transform(img, batch_size=1):
        # 推理数据预处理
        sized = cv2.resize(img, (224, 224))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        sized = ((sized / 255) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        sized = np.transpose(sized, (2, 0, 1))
        sized = sized.astype(np.float32)
        return torch.from_numpy(np.stack([sized] * batch_size))

    def get_device(self):
        return torch.device('cpu')

    def quantification(self, dtype='int8'):
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
        logging.info("test inference result={}".format(self.inference(request)))

    def inference(self, request):
        image_file = request.get("data")
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = np.asarray(image)

        with torch.no_grad():
            # 推理数据预处理
            input_img = self.image_transform(image)

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
    #def quantification(self, dtype='int8'):
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

    def quantification(self, dtype='int8'):
        test_img = cv2.imread(self.test_img)
        with torch.no_grad():
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]

            # 载入模型（针对用户而言，只需要在此工具中加载模型，就能生成离线模型用于部署
            model = models.resnet50()
            model.load_state_dict(torch.load(self.model_file), strict=False)
            model.eval().float()
            model = model.to(torch.device('cpu'))

            # 量化标志
            model = mlu_quantize.quantize_dynamic_mlu(
                model,
                {'iteration': 1, 'mean': mean, 'std': std, 'data_scale': 1.0, 'perchannel': True, 'use_avg': False, 'firstconv': True},
                dtype=dtype,
                gen_quant=True)

            # 输入数据前处理操作
            input_img = self.image_transform(test_img)

            # 在CPU上运行模型进行量化
            out = model(input_img)
            out2 = self.softmax(out.cpu().numpy())
            self.model_file = '/tmp/resnet50-{}.pth'.format(dtype)
            torch.save(model.state_dict(), self.model_file)
            logging.info("quantification success, saved quantification model file is ".format(self.model_file))

    def warm_up(self):
        test_img = cv2.imread(self.test_img)

        with torch.no_grad():
            self.model = models.resnet50()
            self.model.eval().float()

            # 添加设备信息以及载入量化后的 dtype 权重
            self.model = mlu_quantize.quantize_dynamic_mlu(self.model)
            self.model.load_state_dict(torch.load(self.model_file), strict=False)
            self.model.to(self.device)

            # 推理数据预处理
            input_img = self.image_transform(test_img)

            # MLU生成离线模型（对模型进行融合算子操作，提高性能）
            if USE_FUSION:
                input_img = input_img.to(self.device)
                self.model = torch.jit.trace(self.model, input_img, check_trace=False)
                out = self.model(input_img)
                out2 = self.softmax(out.cpu().numpy())
                logging.info('fusion success, inference result={}'.format(out2))

        # 测试模型推理
        self.test()


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
