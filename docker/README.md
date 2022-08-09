# Docker container for demo
To use this library alongside docker, one has to install [docker](https://docs.docker.com/engine/install/ubuntu/) and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to run the container with GPU.
To build a docker image, run:
```
./build.sh
```
To run the container locally with CUDA support, run:
```
docker run -it -p 8888:8888 --gpus all acleto_demo
```
To run the demo notebook, open the jupyter notebook and choose a file for your task in ./acleto/jupyterlab_demo. Note that currently only jupyter notebooks are supported - demo in jupyter lab isn't tested.