#!/bin/bash

# prepare django
python manage.py makemigrations
python manage.py migrate
python manage.py collectstatic --noinput

# start "local" celery worker
C_FORCE_ROOT=false celery -A cidapp.celery_app worker --pool threads --loglevel info &

python manage.py runserver --host $CAID_HOST

# start django
#uvicorn CarnivoreIDApp.asgi:application \
#    --host $CAID_HOST \
#    --port 8080 \
#    --log-config logging.yaml \
#    --log-level info \
#    --reload


#    --host 0.0.0.0 \
