# Note: We do not need install nccl or cudnn, which were already installed in runtime container.
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

RUN apt update

RUN apt install python3 python3-pip -y

RUN cp /usr/bin/python3 /usr/bin/python

RUN cp $(which pip3) /usr/local/bin/pip

RUN pip install --upgrade pip

RUN pip install sagemaker-containers mxnet-cu100 d2l gluonnlp ipython --user

RUN apt install --assume-yes vim

RUN mkdir -p /opt/ml/code

COPY tutorial/ /opt/ml/code/

RUN ls /opt/ml/code/

WORKDIR /opt/ml/code

RUN chmod +x -R /opt/ml/code/

ENV SAGEMAKER_PROGRAM train

ENV PATH="/opt/ml/code:${PATH}"
