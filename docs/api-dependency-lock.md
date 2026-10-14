# API dependency lock

`src/api/requirements-py314.lock` records the exact package versions of the
tested Linux development API on Python 3.14.8 and Django 5.2.18. It includes
all 153 installed packages, including transitive dependencies and packages
previously installed separately in the Dockerfile. This is a version lock;
it does not pin wheel hashes or operating-system packages.

The shared `src/api/Dockerfile` base installs this full snapshot with `uv pip sync` and checks
dependency consistency with `uv pip check`. There are no subsequent package
installs or upgrades that can change the snapshot.

`requirements.txt` retains the direct dependency specifications and upgrade
ranges. Changing it alone no longer changes either API Docker build.

Both the `dev` and `production` targets use Python 3.14.8 and the same lock.
Development mounts application code and runs `start_dev.sh`; production copies
the application into its image and runs `start.sh`. Both scripts retain the
API Celery worker and scheduler. FGVC is not installed in the API.

Compose selects `dev` in `docker-compose.dev.yml` and `production` in
`docker-compose.yml`. Direct builds use:

```bash
docker build -f src/api/Dockerfile --target dev -t caid/api:dev .
docker build -f src/api/Dockerfile --target production -t caid/api:prod .
```

## Updating the snapshot

Perform dependency updates in a disposable environment on the same Linux/Python
target, including the additional packages listed in the lock. Run dependency
checks, the application test suite, and relevant manual flows before capturing
the new snapshot:

```bash
docker compose -f docker-compose.dev.yml exec -T api_dev uv pip check
docker compose -f docker-compose.dev.yml exec -T api_dev python manage.py test caidapp.app_tests --noinput
docker compose -f docker-compose.dev.yml exec -T api_dev uv pip freeze
```

Save the freeze output in the lock file, preserving its explanatory header.
Rebuild the API image and verify that its package versions match the snapshot.

PostgreSQL upgrade work must use a separate database volume and an independently
verified backup/restore. No PostgreSQL image or data changes are part of this lock.
