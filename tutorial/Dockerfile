FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.4.1-cpu-py3

RUN pip install mxnet-mkl d2l https://github.com/dmlc/gluon-nlp/tarball/master -U --user

COPY *.py /opt/ml/model/code/