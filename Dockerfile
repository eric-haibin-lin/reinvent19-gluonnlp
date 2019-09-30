# Note: We do not need install nccl or cudnn, which were already installed in runtime container.
FROM haibinlin/reinvent:latest

RUN cp /usr/bin/python3 /usr/bin/python

RUN cp /usr/local/bin/pip3 /usr/local/bin/pip

RUN pip install sagemaker-containers --user

RUN pip install d2l gluonnlp ipython --user

RUN apt install --assume-yes vim

RUN mkdir -p /opt/ml/code

COPY tutorial/ /opt/ml/code/

ENV SAGEMAKER_PROGRAM train.py

CMD ["/bin/bash"]
