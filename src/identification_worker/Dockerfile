FROM mjirik/wrid-mlbase:23.05
#FROM nvcr.io/nvidia/pytorch:23.05-py3
#
#ARG DEBIAN_FRONTEND=noninteractive
#
## install linux dependencies
#RUN apt-get update --yes && \
#    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
#    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
#    #   apt-get upgrade --yes && \
#    apt-get install --yes --no-install-recommends \
#    wget \
#    git \
#    ffmpeg \
#    libsm6 \
#    libxext6 \
#    tesseract-ocr \
#    exiftool \
#    libwebp-dev \
#    && \
#    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/resources
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -P /root/resources
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P /root/resources
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
wget https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth  -O /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth

COPY ./src/identification_worker/requirements.txt /root/resources/requirements.txt

# install python dependencies
ENV APP_DIR='/identification_worker'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r /root/resources/requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

RUN echo "------- pip list 1 --------"
RUN pip list

RUN echo "------- SAM (pinned, just to be sure) --------"
RUN pip install git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf

# install torch and flash-attn to match sam3 requirements
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip uninstall -y flash-attn
RUN pip install flash-attn --no-build-isolation

RUN pip install git+https://github.com/facebookresearch/sam3.git@46957e47805eaa273f4aa7bbbd25a88bca9108ce
# SAM3 weights are downloaded at runtime via HF_TOKEN (see download_sam3_checkpoint_if_missing)

#RUN pip install pytz --upgrade
#RUN pip install tzdata --upgrade

RUN echo "------- wildlife-tools (pinned, no-deps) --------"
COPY ./src/identification_worker/wildlife-tools /identification_worker/wildlife-tools
RUN pip install -e /identification_worker/wildlife-tools --no-deps
RUN pip install pytorch_metric_learning==2.8.1
RUN pip install transformers==4.30.2
RUN pip install git+https://github.com/cvg/glue-factory.git@1f56839db2242929960d70f85bfac6c19ef2821c
RUN pip install git+https://github.com/cvg/LightGlue.git@edb2b838efb2ecfe3f88097c5fad9887d95aedad


RUN pip install \
    opencv-python==4.5.5.64 \
    opencv-python-headless==4.5.5.64

RUN pip install numpy==1.26.4

RUN echo "------- pip list 1 --------"
RUN echo pip list
RUN pip list

COPY ./src/identification_worker /identification_worker

# expose ports and run application
WORKDIR $APP_DIR
CMD C_FORCE_ROOT=false celery -A worker.identification_worker worker --pool threads --concurrency 1 --loglevel info --queues identification_worker
