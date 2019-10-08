# Note: We do not need install nccl or cudnn, which were already installed in runtime container.
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.4.1-cpu-py3

#RUN apt update
#
#RUN apt install python3 python3-pip -y
#
#RUN cp /usr/bin/python3 /usr/bin/python
#
#RUN cp $(which pip3) /usr/local/bin/pip
#
#RUN pip install --upgrade pip

RUN pip install mxnet-mkl d2l https://github.com/dmlc/gluon-nlp/tarball/master -U --user

#RUN apt install --assume-yes vim

#RUN mkdir -p /opt/ml/code

COPY *.py /opt/ml/model/code/

#COPY bert/* /

#RUN ls /opt/ml/code/

#WORKDIR /opt/ml/code

#RUN chmod +x -R /opt/ml/code/

#ENTRYPOINT ["python", "serve.py"]

#ENV PATH="/opt/ml/code:${PATH}"
