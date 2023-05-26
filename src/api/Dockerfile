FROM python:3.9.16-slim

COPY ./src/api /api

# install python dependencies
ENV APP_DIR='/api'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# Get Boostrap style
ADD https://bootstrapmade.com/arsha-free-bootstrap-html-template-corporate/download/ Arsha.zip

# expose ports and run application
WORKDIR $APP_DIR
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD bash ./start.sh
