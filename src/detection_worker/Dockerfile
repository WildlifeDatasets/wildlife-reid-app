FROM nvcr.io/nvidia/pytorch:23.05-py3
#FROM python:3.10.13-slim

# install linux dependencies
RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# download weights into resources
RUN mkdir -p /detection_worker/resources && \
    wget http://ptak.felk.cvut.cz/plants/DanishFungiDataset/md_v5a.0.0.pt \
    -O /detection_worker/resources/md_v5a.0.0.pt

RUN wget http://ptak.felk.cvut.cz/plants/DanishFungiDataset/sam_vit_h_4b8939.pth \
    -O /detection_worker/resources/sam_vit_h_4b8939.pth

COPY ./src/detection_worker/requirements.txt /detection_worker/requirements.txt

# install python dependencies
ENV APP_DIR='/detection_worker'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build setuptools && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

COPY ./src/detection_worker /detection_worker

RUN pip install pytz --upgrade
RUN pip install tzdata --upgrade

# expose ports and run application
WORKDIR $APP_DIR
CMD C_FORCE_ROOT=false celery -A worker.detection_worker worker --pool threads --loglevel info --queues detection_worker
