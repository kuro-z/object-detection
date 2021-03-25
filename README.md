# 基于Flask的YOLOv5部署Demo

文件结构：

- detector.py 模型初始化、数据处理和结果返回等集成
- server.py flask服务，定义路由等。启动时会生成一系列用于缓存的文件夹
- fake_client.py 给出了调用API的样例

- clear_caches.py 清空缓存

- config.yaml 配置文件，预设置参数
- models、utils、weights用于yolo必要的模型文件等
- templates web主页
- static 存放js、css和访问资源等

启动服务

```shell
python server.py
```

