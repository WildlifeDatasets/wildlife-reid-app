FROM python:3.10.13-slim

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

COPY ./src/identification_worker/requirements.txt /identification_worker/requirements.txt

# install python dependencies
ENV APP_DIR='/identification_worker'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

COPY ./src/identification_worker /identification_worker

# expose ports and run application
WORKDIR $APP_DIR
CMD C_FORCE_ROOT=false celery -A worker.identification_worker worker --pool threads --loglevel info --queues identification_worker
