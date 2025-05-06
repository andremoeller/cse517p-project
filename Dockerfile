# slim inference runtime image -- should be built fast
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS runtime
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

RUN apt update && apt install -y build-essential # triton needs gcc

COPY requirements.txt /job/requirements.txt
RUN pip install -r /job/requirements.txt

COPY . /job/

# development image -- this is the one devcontainer.json uses.
# we can put a kitchen sink here.
FROM runtime AS dev

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    less \
    htop \
    vim \
    zip


COPY requirements_dev.txt /job/requirements_dev.txt

RUN pip install -r /job/requirements_dev.txt
