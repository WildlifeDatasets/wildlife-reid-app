FROM python:3.9.16-slim

COPY ./src/api /api

# install python dependencies
ENV APP_DIR='/api'
WORKDIR $APP_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# expose ports and run application
WORKDIR $APP_DIR
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD python manage.py makemigrations && \
    python manage.py migrate && \
    python manage.py collectstatic --noinput && \
    uvicorn CarnivoreIDApp.asgi:application \
    --host 0.0.0.0 \
    --port 8080 \
    --log-config logging.yaml \
    --log-level info
