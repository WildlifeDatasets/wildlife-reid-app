# Configuration Guide

The application configuration is managed primarily through environment variables loaded from `.env` files. This allows for separation between code and configuration, making it easier to deploy across different environments (development, production).

## Environment Files

The Docker Compose configuration loads variables from several sources in the following priority:

1.  `.env` (Project root - **User defined secrets**)
2.  `src/variables.prod.env` (Production defaults)
3.  `src/variables.env` (Shared defaults)
4.  `src/variables.dev.env` (Development overrides)

**Important:** You should create a `.env` file in the root directory `wildlife-reid-app/` to store sensitive keys and local path configurations. This file is ignored by Git.

## Key Configuration Variables

### General Settings

| Variable    | Description                                                     |  Example                                   |
|:------------|:----------------------------------------------------------------|:-------------------------------------------|
| `CAID_HOST` | The hostname or IP address where the application is accessible. | `localhost`, `192.168.1.50`, `example.com` |
| `DEBUG`     | Enables Django debug mode. **Set to `False` in production.**    | `True` / `False`                           |
| `TZ`        | Timezone setting for containers.                                | `Europe/Prague`                            |

### Authentication & Integrations

| Variable | Description |
| :--- | :--- |
| `ALLAUTH_GOOGLE_CLIENT_ID` | Google OAuth Client ID for social login. |
| `ALLAUTH_GOOGLE_CLIENT_SECRET` | Google OAuth Client Secret. |
| `WANDB_API_KEY` | API Key for Weights & Biases (used for experiment tracking in workers). |

### Data & Storage

| Variable | Description |
| :--- | :--- |
| `CAID_IMPORT` | Local path on the host machine to mount for bulk data imports. Defaults to `./caid_import`. |
| `SFTP_CMD` | Command for mounting external storage via SSHFS (if used). |
| `SFTP_PASSWORD` | Password for the SFTP mount. |

### Database Credentials

These variables control the connection to the PostgreSQL databases.

*   `POSTGRES_DB`
*   `POSTGRES_USER`
*   `POSTGRES_PASSWORD`
*   `POSTGRES_HOST`
*   `POSTGRES_PORT`

### Worker Resources

| Variable | Description |
| :--- | :--- |
| `API_IMAGE` | Docker image tag for the API. |
| `TAXON_WORKER_IMAGE` | Docker image tag for the Taxon Worker. |
| `IDENTIFICATION_WORKER_IMAGE` | Docker image tag for the ID Worker. |

## Development vs. Production

### Development (`docker-compose.dev.yml`)
*   Uses `src/variables.dev.env`.
*   Sets `DEBUG=True`.
*   Mounts local source code directories (`./src/api`, etc.) into containers for live code updates.
*   Exposes ports directly for easier debugging.

### Production (`docker-compose.yml`)
*   Uses `src/variables.prod.env`.
*   Should have `DEBUG=False`.
*   Uses built Docker images rather than local code mounts.
*   Restarts containers automatically (`restart: always`).
