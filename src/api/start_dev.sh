#!/bin/bash

# prepare django
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --noinput

# start "local" celery worker
C_FORCE_ROOT=false celery -A caidapp.celery_app worker --pool threads --loglevel info &


# start django
#python manage.py runserver $CAID_HOST:8080
uvicorn CarnivoreIDApp.asgi:application \
    --host 0.0.0.0 \
    --port 8080 \
    --log-config logging.yaml \
    --log-level info \
    --reload


    # --host $CAID_HOST \
