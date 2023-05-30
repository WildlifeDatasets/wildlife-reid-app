FROM python:3.9.16-slim

COPY ./src/api /api

# install python dependencies
ENV APP_DIR='/api'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# Get Boostrap style
ENV CAID_STATIC='/caid_static'
ENV CAID_STATIC='/api/static'
RUN mkdir -p $CAID_STATIC
WORKDIR $CAID_STATIC
RUN apt-get update && apt-get install -y wget unzip

RUN wget https://bootstrapmade.com/content/templatefiles/NiceAdmin/NiceAdmin.zip
RUN unzip NiceAdmin.zip
RUN mkdir -p $CAID_STATIC
RUN mv  NiceAdmin/assets $CAID_STATIC

# expose ports and run application
WORKDIR $APP_DIR
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD bash ./start.sh
