FROM python:3.14.8-slim AS base

# install system dependencies
#RUN sudo add-apt-repository ppa:ubuntugis/ppa
RUN apt-get update --yes &&  \
    apt-get install --yes \
    wget \
    unzip \
    ffmpeg \
    curl \
    mc \
    gdal-bin && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# This approximately follows this guide: https://hynek.me/articles/docker-uv/
# Which creates a standalone environment with the dependencies.
# - Silence uv complaining about not being able to use hard links,
# - tell uv to byte-compile packages for faster application startups,
# - prevent uv from accidentally downloading isolated Python builds,
# - pick a Python (use `/usr/bin/python3.12` on uv 0.5.0 and later),
# - and finally declare `/app` as the target for `uv sync`.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/code/.venv

COPY --from=ghcr.io/astral-sh/uv:0.5.2 /uv /uvx /bin/

RUN uv venv /code/.venv
ENV PATH="/code/.venv/bin:$PATH"

COPY ./src/api/requirements-py314.lock /tmp/requirements-py314.lock

# install python dependencies
ENV APP_DIR='/api'
WORKDIR $APP_DIR
RUN uv pip sync --no-cache-dir --compile /tmp/requirements-py314.lock && \
    uv pip check && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

RUN python -c "import django; assert django.VERSION[:2] == (5, 2), django.get_version()"

ENV PYTHONUNBUFFERED=1

FROM base AS dev
COPY ./src/api/start_dev.sh /api/start_dev.sh
EXPOSE 22280
CMD ["bash", "./start_dev.sh"]

FROM base AS production
COPY ./src/api /api
# Keep collected assets outside /shared_data: that path is mounted over SSHFS
# at runtime. Docker caches this layer until application sources change.
ENV DJANGO_STATIC_ROOT=/opt/caid-static
RUN mkdir -p /data && \
    DEBUG=False SHARED_DATA_PATH=/tmp/static-build PRIVATE_DATA_PATH=/tmp/static-build \
    POSTGRES_DB=static_build POSTGRES_HOST=unused POSTGRES_PORT=5432 \
    POSTGRES_USER=unused POSTGRES_PASSWORD=unused \
    RABBITMQ_URL=memory:// REDIS_URL=cache+memory:// \
    python manage.py collectstatic --noinput --verbosity 1 --ignore '*css-b64-images*test*'
EXPOSE 8080
CMD ["bash", "./start.sh"]
