# Wildlife-ReID-App

## Run Application
The application is orchestrated using `docker-compose` - a tool for defining and running multi-container Docker application.

Before starting the application create `.env` file with secret variables.
```bash
echo "WANDB_API_KEY=..." >> .env
echo "HF_TOKEN=..." >> .env  # Hugging Face token for SAM3
echo "ALLAUTH_GOOGLE_CLIENT_ID=..." >> .env
echo "ALLAUTH_GOOGLE_CLIENT_SECRET=..." >> .env
echo "CAID_HOST=147.228..." >> .env
```

```bash
export CAID_HOST="147.228..."
```

Optionally, you can add `DATA_IMPORT_DIR` to your environment variables.

```bash
echo "CAID_IMPORT=/mnt/caid_import" >> .env
```


```bash
docker compose up --build -d
```

or restart existing containers:
```bash
date && docker compose down && git pull && docker compose up -d --build && date
```


## Advanced setup

### Import directory

Optionally, you can add `DATA_IMPORT_DIR` to your environment variables.
Set project name (default is dir name) to shorten container name and distinguish between development and production.
```bash
echo "CAID_IMPORT=/mnt/caid_import" >> .env
echo "COMPOSE_PROJECT_NAME=caid_local" >> .env
```

## Production health status check


To check whether the application is running, you can use the following command to check the health status of the containers:
    1) Create account `system_healthcheck` and add ZIP files with mediafiles with various species (in taxon processing), 
        known identities (re-id database), and unknown identities (re-id query).
    2) Check manually the health status of the containers:
         ```bash
         docker compose exec api python manage.py healthcheck_inference
         ```
    3) Create `cron` job to check the health status of the containers every 4 hours:
         ```bash
         sudo touch /var/log/caid_healthcheck.log
         sudo chmod 664 /var/log/caid_healthcheck.log
         sudo crontab -e
         0 */4 * * * docker compose -f /home/myuser/Projects/CarnivoreID-App/docker-compose.yml exec api python manage.py healthcheck_inference >> /var/log/caid_healthcheck.log 2>&1
         ```


## Identification dashboard statistics

The API startup scripts launch Celery Beat alongside the API worker. The schedule
is defined in `src/api/caidapp/celery.py` under `app.conf.beat_schedule`: every
three hours it publishes `caidapp.tasks.refresh_identification_dashboard_stats`
to the `celery` queue. This task dispatches one refresh per workgroup to the API
worker. Failed refreshes retry independently up to three times; existing snapshots
remain available. When the worker starts, it also queues a refresh of missing or
stale snapshots without waiting for the first periodic run.

The dashboard displays the snapshot timestamp; running task status and action
availability still use current data. A workgroup without a snapshot is calculated
once on its first dashboard request if the worker has not prepared it yet.

After deploying, apply migrations and restart the API to start Beat and the worker.
Run only one Beat instance for the deployment. Its schedule state is persisted in
`/data/celerybeat-schedule`. If launching the API without the startup scripts,
start `celery -A caidapp.celery_app beat` alongside the API Celery worker.

The management command remains available for a manual, immediate refresh
(optionally limited to one workgroup):

```bash
docker compose exec api python manage.py refresh_identification_dashboard_stats --workgroup-id 1
```

## Build `wrid-mlbase` image

```bash
docker build -t mjirik/wrid-mlbase:23.05 -f Dockerfile.wrid-mlbase .
```


### Development
Run the following commands to build and start the application in the development mode.


```bash
docker compose -f docker-compose.dev.yml up -d --build
```

Run the following commands to view the final development mode configuration with overrides from `docker-compose.dev.yml`. 
```bash
docker compose -f docker-compose.dev.yml config
```

Create superuser:
```bash
docker compose -f docker-compose.dev.yml exec api_dev python manage.py createsuperuser
```

In admin panel create new Workgroup and then in `ciduser` add this workgroup to user.

Make migrations and migrate, if needed:
```bash
docker exec -it carnivoreid-app-dev-api bash -ic 'python manage.py makemigrations'
docker exec -it carnivoreid-app-dev-api bash -ic 'python manage.py migrate'
```


## Build initial ML docker image

```bash
docker build -t mjirik/wrid-mlbase:23.05 -f Dockerfile.wrid-mlbase . 
```


### Run tests

```bash 
docker compose -f docker-compose.dev.yml exec api_dev python manage.py test
```

Run the default fast subset and exclude tests tagged as `long`:

```bash
docker compose -f docker-compose.dev.yml exec api_dev python manage.py test_fast
```

Run only long Django tests:

```bash
docker compose -f docker-compose.dev.yml exec api_dev python manage.py test --tag=long
```

Run the normal Django suite but exclude long tests explicitly:

```bash
docker compose -f docker-compose.dev.yml exec api_dev python manage.py test --exclude-tag=long
```

### New upload smoke test

The new upload entry is shown to users who can use taxon classification, re-identification, or who have admin access.
Inside the form:

- users with only taxon classification upload directly to taxon processing,
- users with only re-id upload directly to re-identification,
- users with both choose between taxon processing and re-id,
- only users with base-dataset permission can mark a re-id upload as identified individuals for the base dataset.

To run the real new-upload integration healthcheck in development, mount your local test dataset into the dev containers:

```bash
echo "WRAP_TEST_DATA_HOST_DIR=C:/Users/mjirik/syno_bmc_home/my_bc_data/data/biology/orig/CarnivoreID" >> .env
```

Then rebuild the dev stack and run:

```bash
docker compose -f docker-compose.dev.yml exec api_dev python manage.py healthcheck_new_upload
```

This command uploads `2021-05-06_Tri_lokality_XYZ.zip` through the new upload endpoint, waits for async processing,
and verifies that a new locality `Xandovice` is created for the dedicated healthcheck user.
Unlike `manage.py test`, this is a live smoke check against the running development stack and its normal dev database.

### Sample data

The sample data can be added by creating `ArchiveCollection` with name `sample_data` and selection of several 
`UploadedArchive` instances into this collection.


# License

This project uses Annotorious (https://recogito.github.io/annotorious/),
licensed under the BSD 3-Clause License.
