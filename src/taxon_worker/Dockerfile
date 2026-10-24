FROM mjirik/wrid-mlbase:23.05
# FROM nvcr.io/nvidia/pytorch:23.05-py3

ARG DEBIAN_FRONTEND=noninteractive

# Previously supplied by mjirik/wrid-mlbase:23.05; keep the worker self-contained
# until a new shared base is validated for both ML workers.
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    wget \
#    git \
#    ffmpeg \
#    libsm6 \
#    libxext6 \
#    tesseract-ocr \
#    exiftool \
#    libwebp-dev \
# && rm -rf /var/lib/apt/lists/*

# The old NVIDIA base configures an unavailable extra pip index in several files.
# Recheck this workaround when upgrading to a newer PyTorch/CUDA base image:
# it may no longer be needed there. Upgrading the torch wheel alone will not fix it.
# Keep PyPI, other settings, and explicit --index-url options intact.
RUN python - <<'PY'
from configparser import ConfigParser
from pathlib import Path

for filename in (
    "/etc/xdg/pip/pip.conf",
    "/etc/pip.conf",
    "/usr/pip.conf",
    "/root/.pip/pip.conf",
    "/root/.config/pip/pip.conf",
):
    path = Path(filename)
    if not path.is_file():
        continue
    config = ConfigParser(interpolation=None)
    config.read(path)
    changed = False
    for section in config.sections():
        for option in ("extra-index-url", "trusted-host"):
            if not config.has_option(section, option):
                continue
            values = config.get(section, option).split()
            kept = [value for value in values if value.rstrip("/") not in (
                "https://pypi.ngc.nvidia.com", "http://pypi.ngc.nvidia.com", "pypi.ngc.nvidia.com",
            )]
            if kept != values:
                changed = True
                if kept:
                    config.set(section, option, "\n".join(kept))
                else:
                    config.remove_option(section, option)
    if changed:
        with path.open("w") as output:
            config.write(output)
PY

# MegaDetector weights and YOLOv5 Hub code are cached in /shared_data at runtime.
RUN pip install pillow==10.3.0 tqdm==4.66.3 opencv-python-headless==4.6.0.66

# install python dependencies
ENV APP_DIR='/taxon_worker'
WORKDIR $APP_DIR
#RUN pip install --no-cache-dir --upgrade pip build && \
#    pip install --no-cache-dir --compile timm==0.9.12 && \
#    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*
#RUN pip install https://github.com/BohemianVRA/FGVC/releases/download/v1.5.4-dev/fgvc-1.5.4.dev0-py3-none-any.whl

# install torch and flash-attn to match sam3 requirements
RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip uninstall -y flash-attn
RUN pip install flash-attn --no-build-isolation
RUN pip install git+https://github.com/facebookresearch/sam3.git@46957e47805eaa273f4aa7bbbd25a88bca9108ce
# SAM3 weights are downloaded at runtime via HF_TOKEN (see download_sam3_checkpoint_if_missing)
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

COPY ./src/taxon_worker/requirements.txt /root/resources/requirements.txt

#RUN pip install pytz --upgrade
#RUN pip install tzdata --upgrade

RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r /root/resources/requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# RUN pip install --no-cache-dir --upgrade pip build && \
#     pip install \
#     opencv-python==4.5.5.64 \
#     opencv-python-headless==4.5.5.64 && \
#     rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*
#
# RUN pip install timm==0.6.13

RUN pip install numpy==1.26.4

RUN python - <<EOF
import timm
timm.create_model(
    "hf-hub:strakajk/Lynx-Orientation-ResNet10t-176",
    pretrained=True,
    num_classes=4
)
EOF

COPY ./src/taxon_worker /taxon_worker

RUN pip uninstall -y opencv-python opencv-python-headless opencv opencv-contrib-python cv2 && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2* && \
    rm -rf /usr/local/lib/python3.10/site-packages/cv2* && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Download EasyOCR models during image build so the first runtime OCR task does not block on model fetching.
RUN python -c "import easyocr; easyocr.Reader(['en']); print('easyocr models loaded')"

# expose ports and run application
WORKDIR $APP_DIR
CMD C_FORCE_ROOT=false celery -A worker.taxon_worker worker --pool threads --concurrency 1 --loglevel info --queues taxon_worker
