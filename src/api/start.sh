#!/bin/bash

# get Boostrap style
STATIC_DIR="$SHARED_DATA_PATH/static"
if [ ! -d "$STATIC_DIR/assets" ]; then
  echo 'Loading NiceAdmin bootstrap assets.'
  wget https://bootstrapmade.com/content/templatefiles/NiceAdmin/NiceAdmin.zip -P $STATIC_DIR && \
    unzip "$STATIC_DIR/NiceAdmin.zip" -d "$STATIC_DIR" && \
    mv "$STATIC_DIR/NiceAdmin/assets" "$STATIC_DIR"
fi

# prepare django
# python manage.py makemigrations --noinput --verbosity 2
python manage.py migrate --noinput --verbosity 2
python manage.py collectstatic --noinput --verbosity 2

# start "local" celery worker
C_FORCE_ROOT=false celery -A caidapp.celery_app worker --pool threads --concurrency 4 --loglevel info &

# start django
uvicorn CarnivoreIDApp.asgi:application \
    --host 0.0.0.0 \
    --port 8080 \
    --log-config logging.yaml \
    --log-level info
