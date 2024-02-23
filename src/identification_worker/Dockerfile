FROM nvcr.io/nvidia/pytorch:23.05-py3

ARG DEBIAN_FRONTEND=noninteractive

# install linux dependencies
RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    # apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./src/identification_worker/requirements.txt /identification_worker/requirements.txt

# install python dependencies
ENV APP_DIR='/identification_worker'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

COPY ./src/identification_worker /identification_worker

RUN pip install \
    opencv-python==4.5.5.64 \
    opencv-python-headless==4.5.5.64

pip install pytz --upgrade
pip install tzdata --upgrade

# expose ports and run application
WORKDIR $APP_DIR
CMD C_FORCE_ROOT=false celery -A worker.identification_worker worker --pool threads --loglevel info --queues identification_worker
