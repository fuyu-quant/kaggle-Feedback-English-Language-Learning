FROM ubuntu:20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && \
    apt-get install -yq --no-install-recommends python3-pip \
        build-essential \
        python3-dev \
        wget \
        git  \
        tzdata && apt-get upgrade -y && apt-get clean

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --no-cache-dir torch \
        #matplotlib \
        #sklearn \
        sagemaker-training \
        transformers \
        sentencepiece \
        tqdm \
        pyyaml \
        hydra-core \
        wandb \
        pandas
        