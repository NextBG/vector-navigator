FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

WORKDIR /usr/src/app

RUN pip install -r requirements.txt

COPY ./ .