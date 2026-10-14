# Django 5.2 upgrade

Branch: `mji/dja52`. This stage upgrades the API framework without changing the
Python Docker image, Celery, the ML workers, or the observation domain model.

## Changes

- Django 4.2.16 → 5.2.18.
- django-allauth 0.54.0 → 65.19.7, with the `socialaccount` extra and required
  `AccountMiddleware`. Existing Google provider settings and login behavior remain.
- Replace the obsolete `django-rest-framework` wrapper with
  `djangorestframework==3.16.1`.
- Remove unused `django-cruds==3.0.0`, which requires Django < 5.
- Replace `STATICFILES_STORAGE` with `STORAGES`, preserving the development
  storage, production WhiteNoise manifest storage, and default media storage.
- Exclude only css-b64-images test fixtures bundled by handyhelpers during
  `collectstatic`. Their CSS references an image absent from the package.
- Add tests for the login page, Google OAuth redirect parameters, password login
  and logout, and rejection of invalid passwords. These do not contact Google.

## Local verification (2026-10-09)

The Docker daemon was unavailable. Runtime checks used Python 3.11.14 and an
in-memory SQLite database, with local data paths and in-memory Celery configuration.
The separate Django 5.2 environment reused the existing local test environment's
other dependencies read-only; this is not a clean Docker requirements installation.

- API requirements resolved successfully for target Python 3.10.
- `manage.py check`: no issues.
- `manage.py makemigrations --check --dry-run`: no model changes.
- Fresh test database migrations, including updated allauth migrations: succeeded.
- Production `collectstatic` with the fixture exclusion: succeeded.
- Django 4.2 baseline: 428 tests, 4 failures, 2 errors, 1 skipped.
- Django 5.2: 432 tests, the same 4 failures and 2 errors, 1 skipped.
  All four new authentication tests passed.

The existing unsuccessful tests are:

- `IdentificationOutlierSuggestionsCallbackTest.test_success_callback_resolves_candidate_mediafile_from_worker_path`
- `MediaFileLocationFallbackTest.test_prepare_mediafile_for_identification_resolves_masked_reference_paths`
- `IdentityCodeSuggestionsViewTest.test_status_endpoint_returns_progress_meta`
- `UrlSmokeTest.test_all_named_caidapp_urls`
- `IdentificationRerunTest.test_download_init_identification_csv_returns_latest_workgroup_csv`
- `IdentificationRerunTest.test_download_run_identification_csv_returns_latest_workgroup_csv`

## Before deployment

Build the API development image and run checks against a disposable PostgreSQL
database. From the repository root, with the development database/broker available:

```bash
docker compose -f docker-compose.dev.yml build api_dev
docker compose -f docker-compose.dev.yml run --rm --no-deps api_dev python manage.py check
docker compose -f docker-compose.dev.yml run --rm --no-deps api_dev python manage.py makemigrations --check --dry-run
docker compose -f docker-compose.dev.yml run --rm --no-deps api_dev python manage.py migrate --plan
docker compose -f docker-compose.dev.yml run --rm --no-deps api_dev python -Wd manage.py test caidapp.app_tests
```

Review and test the allauth migration plan on a disposable copy of the deployment
database. The fresh SQLite migration check does not validate existing account data.
Back up the deployment database before applying migrations.

Check actual Google login and callback in staging, including the configured HTTPS
callback URL; local tests verify only initiation of OAuth. Also verify media upload,
observation editing, imports/exports, permissions, and Celery worker callbacks.

The production Docker build (including additional packages installed directly in
its Dockerfile), PostgreSQL validation, and live OAuth remain to be verified.
Upgrade the Python API image to 3.12 as a subsequent change.
