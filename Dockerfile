# hash:sha256:de7a4761cb2a86db16ca7602535edf58e144a21929b974c1edab9de20a40624c
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-python3.7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    keras==2.3.1 \
    matplotlib==3.0.2 \
    openpyxl==3.0.6 \
    pandas==1.2.2 \
    scikit-learn==0.24.1 \
    tensorflow==1.13.1
