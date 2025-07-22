FROM nvcr.io/nvidia/pytorch:23.05-py3

ARG DEBIAN_FRONTEND=noninteractive

# install linux dependencies
RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    #   apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    tesseract-ocr \
    exiftool \
    && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/resources
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P /root/resources

COPY ./src/identification_worker/requirements.txt /identification_worker/requirements.txt

# install python dependencies
ENV APP_DIR='/identification_worker'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

RUN echo "------- pip list 1 --------"
RUN pip list

RUN pip install git+https://github.com/facebookresearch/segment-anything.git

RUN pip install pytz --upgrade
RUN pip install tzdata --upgrade

COPY ./src/identification_worker /identification_worker
RUN pip install -e /identification_worker/wildlife-tools --no-deps
RUN pip install transformers==4.30.2
RUN pip install git+https://github.com/cvg/glue-factory.git@1f56839db2242929960d70f85bfac6c19ef2821c


RUN pip install \
    opencv-python==4.5.5.64 \
    opencv-python-headless==4.5.5.64
RUN echo "------- pip list 1 --------"
RUN echo pip list
RUN pip list

# expose ports and run application
WORKDIR $APP_DIR
# CMD C_FORCE_ROOT=false celery -A worker.identification_worker worker --pool threads --concurrency 1 --loglevel info --queues identification_worker
