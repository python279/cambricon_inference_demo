此demo实现了torch resnet50在CPU/GPU/MLU270上的推理功能，封装了TorchResnet50InferenceBase这个类，定义以下几个接口，get_device（获取对应的计算设备）、quantification（量化）、warm_up（加载模型，预热，MLU270融合）、inference（推理），使用flask作为web框架，并且集成了gunicorn实现多进程提高并发度。

实际测试的时候，gunicorn在MLU270设备上的损耗比较大，目前还不确定原因，建议直接python Service.py运行flask服务。

启动服务：
```
sh run.sh
```

测试接口：
```
python test.py
```