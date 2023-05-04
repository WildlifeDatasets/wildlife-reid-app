FROM python:3.9.16-slim

COPY ./src/inference_worker /inference_worker

# install python dependencies
ENV APP_DIR='/inference_worker'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# expose ports and run application
WORKDIR $APP_DIR
USER 1000
CMD celery -A worker worker --pool threads --loglevel info --queues inference_worker
