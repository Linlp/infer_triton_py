1、开发环境
pip install paddleocr==2.7.0
pip install tritonclient onnxruntime gevent geventhttpclient

2、启动tritonserver容器和服务
docker run -it --gpus=all --net=host -v ./models:/models nvcr.io/nvidia/tritonserver:24.07-py3
tritonserver --model-repository=./gd_models/ --log-verbose=1