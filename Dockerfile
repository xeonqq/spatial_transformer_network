FROM tensorflow/tensorflow:nightly-jupyter
MAINTAINER Qian Qian (xeonqq@gmail.com)

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN pip install imgaug
