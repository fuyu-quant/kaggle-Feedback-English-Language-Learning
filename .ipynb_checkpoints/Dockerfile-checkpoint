# GPUインスタンス用のimage
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && \
    apt-get install -yq --no-install-recommends python3-pip \
        build-essential \
        python3-dev \
        wget \
        unzip \
        git  \
        tzdata && apt-get upgrade -y && apt-get clean

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --no-cache-dir transformers \
        sagemaker-training \
        sentencepiece \
        tqdm \
        pyyaml \
        hydra-core \
        wandb \
        pandas \
        argparse
        