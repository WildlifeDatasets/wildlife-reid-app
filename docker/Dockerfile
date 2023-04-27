FROM nvidia/cuda:11.3.1-cudnn8-devel
ENV TZ="Europe/Berlin"
#FROM nginx
MAINTAINER Miroslav Jiřík <mjirik@kky.zcu.cz>
EXPOSE 8000

ENV DEBIAN_FRONTEND noninteractive
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update
RUN apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo ninja-build
RUN ln -sv /usr/bin/python3 /usr/bin/python

## create a non-root user
#ARG USER_ID=1000
#RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER appuser
##WORKDIR /home/appuser
WORKDIR /webapps

#ENV PATH="/home/appuser/.local/bin:${PATH}"
#RUN wget https://bootstrap.pypa.io/get-pip.py && \
#	python3 get-pip.py --user && \
#	rm get-pip.py


#####################

#VOLUME .. /
USER root
SHELL ["/bin/bash", "--login", "-c"]
RUN #apt-get update --yes
# PyQt5 => ffmpeg libsm6 libxext6
RUN apt-get install --yes gpg vim redis ffmpeg libsm6 libxext6 curl supervisor multitail
# Install our public GPG key to trusted store
RUN curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > conda.gpg
RUN install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg

# Check whether fingerprint is correct (will output an error message otherwise)
RUN gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg --no-default-keyring --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806

# Add our Debian repo
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list
RUN apt-get update --yes
RUN apt-get install --yes conda

# make conda visible
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc
RUN /opt/conda/condabin/conda init bash
# RUN source /opt/conda/etc/profi le.d/conda.sh
ENV PATH="/opt/conda/condabin:${PATH}"
ARG PATH="/opt/conda/condabin:${PATH}"


# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER appuser

# COPY scaffanweb .
COPY environment.yml .
COPY requirements_pip.txt .

RUN conda env create -f environment.yml
RUN conda list
#RUN cd /webapps/scaffanweb_django
# Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "scaffanweb", "/bin/bash", "--login", "-c"]
RUN /opt/conda/condabin/conda init bash
RUN conda run -n cidapp--no-capture-output pip install -r requirements_pip.txt

# gunicorn is in conda env
#RUN conda run -n cidapp--no-capture-output pip install gunicorn

#CMD conda run -n cidapp--no-capture-output python manage.py qcluster
#CMD service redis-server start && tail -F /var/log/redis/error.log

# Install Detectron2 dependencies
#################################

# See https://pytorch.org/ for other options if you use a different version of CUDA
#RUN conda run -n cidapp--no-capture-output pip install tensorboard cmake   # cmake from apt-get is too old
#RUN conda run -n cidapp--no-capture-output pip install torch==1.9 torchvision==0.10 -f https://download.pytorch.org/whl/cu111/torch_stable.html
#
#RUN conda run -n cidapp--no-capture-output pip install 'git+https://github.com/facebookresearch/fvcore'
## install detectron2
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
## set FORCE_CUDA because during `docker build` cuda is not accessible
#ENV FORCE_CUDA="1"
## This will by default build detectron2 for all common cuda architectures and take a lot more time,
## because inside `docker build`, there is no way to tell which architecture will be used.
#ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
#ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
#
#RUN conda run -n cidapp--no-capture-output pip install -e detectron2_repo
#
## Set a fixed model cache directory.
#ENV FVCORE_CACHE="/tmp"

######### - Detectron 2 install end
#WORKDIR /home/appuser/detectron2_repo



WORKDIR /webapps

RUN conda run -n cidapp--no-capture-output pip install django-q

#RUN apt-get install multitail
RUN mkdir -p ~/piglegcv/logs/

# for dash conversion
# RUN apt-get install --yes gpac

# remove files to fix E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64  InRelease' is no longer signed.
# https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN sudo rm /etc/apt/sources.list.d/cuda.list
RUN sudo rm /etc/apt/sources.list.d/nvidia-ml.list

# to get rotatelogs
RUN sudo apt-get update --yes
RUN sudo apt-get install --yes apache2-utils


# RUN sudo chmod a+w /webapps/CarnivoreID-App/cidapp/private/db.sqlite3
#ls -l /webapps/CarnivoreID-App/cidapp/private/

#RUN conda install -n cidapp-c conda-forge django-environ
#COPY start.sh
#CMD cd /webapps/CarnivoreID-App/cidapp && conda run -n cidapp--no-capture-output python manage.py qcluster
#CMD service redis-server start && tail -F /var/log/redis/error.log
#CMD tail -f /dev/null

#ENV PIGLEGCV_HOSTNAME "PIGLEGCV_HOSTNAME"
CMD cd /webapps/CarnivoreID-App/docker && \
    bash ./start.sh && \
    tail -f /dev/null
#CMD tail -f /dev/null

#    conda run -n cidapp--no-capture-output python manage.py qcluster && \
#    conda run -n cidapp--no-capture-output python manage.py runserver 0:8000 && \
