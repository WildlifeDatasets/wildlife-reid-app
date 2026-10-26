#!/bin/bash

# prepare django
# python manage.py makemigrations --noinput --verbosity 2
python manage.py migrate --noinput --verbosity 2
# Production static files are collected into the image during docker build.

# start "local" celery worker
C_FORCE_ROOT=false celery -A caidapp.celery_app worker --pool threads --concurrency 12 --loglevel info &

# Run one scheduler for the API deployment; it dispatches tasks to the API worker.
celery -A caidapp.celery_app beat --loglevel info --schedule /data/celerybeat-schedule --pidfile /tmp/caid-celerybeat.pid &

# start django
uvicorn CarnivoreIDApp.asgi:application \
    --host 0.0.0.0 \
    --port 8080 \
    --log-config logging.yaml \
    --log-level info
