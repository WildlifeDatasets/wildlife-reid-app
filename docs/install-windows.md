# Installation

This app is designed to be run via **Docker Compose**.

## Prerequisites

- **Git**
- **Docker Engine** + **Docker Compose v2** (`docker compose ...`)

## 1. Clone the repository

```bash
git clone https://github.com/WildlifeDatasets/wildlife-reid-app.git
cd wildlife-reid-app
```

## 2. Create `.env` with required secrets

Create a `.env` file in the repository root and add the required variables:

```bash
echo "WANDB_API_KEY=..." >> .env
echo "ALLAUTH_GOOGLE_CLIENT_ID=..." >> .env
echo "ALLAUTH_GOOGLE_CLIENT_SECRET=..." >> .env
echo "CAID_HOST=147.228..." >> .env
```
If you are creating the file manually, make sure that the line endings are set to LF (Unix), not CRLF, otherwise Docker scripts will not understand them.

The README also exports `CAID_HOST` in the shell (optional, depending on your workflow):

```bash
export CAID_HOST="147.228..."
```

### Optional: import directory

Optionally, add an import directory variable:

```bash
echo "CAID_IMPORT=/mnt/caid_import" >> .env
```

(Adjust paths to your environment.)

## 3. Download and install NiceAdmin static assets

The repository expects NiceAdmin assets under `src/api/static/`:

Download the source files directly from this link:
```bash
https://github.com/hacktheme/Nice-Admin
```
Open the downloaded NiceAdmin-master.zip and locate the `assets` folder inside.

Copy the entire assets folder into your project directory at:
```bash
wildlife-reid-app/src/api/static/
```
(If the `static` folder does not exist yet, create it manually inside `src/api/`).

## 4. Run in development mode

Now we will create container "images." This step is the most time-consuming and requires the most internet bandwidth.
```bash
docker compose -f docker-compose.dev.yml build
```
This step takes a long time (15–30 minutes). Big AI libraries such as OpenCV or PyTorch (hundreds of MB) are downloaded. If the process seems to be stuck at "pip install," that's okay, it just takes time and a stable internet connection.

After building, you need to run the application with this command:
```bash
docker compose -f docker-compose.dev.yml up
```
To see the fully expanded compose configuration:

```bash
docker compose -f docker-compose.dev.yml config
```

## 5. First-time setup (Django admin + DB)

### Create a superuser

```bash
docker exec -it carnivoreid-app-dev-api bash -ic 'python manage.py createsuperuser'
```

After that, in the **admin panel**:

- Create a new **Workgroup**
- Then, in `ciduser`, add this workgroup to the user

### (If needed) Make migrations and migrate

```bash
docker exec -it carnivoreid-app-dev-api bash -ic 'python manage.py makemigrations'
docker exec -it carnivoreid-app-dev-api bash -ic 'python manage.py migrate'
```

## 6. Run tests

```bash
docker compose -f docker-compose.dev.yml exec api_dev python manage.py test
```

## 7. Sample data (optional)

To add sample data:

- Create an `ArchiveCollection` named `sample_data`
- Select multiple `UploadedArchive` instances and add them into that collection

## License note

This project uses **Annotorious**, licensed under the **BSD 3-Clause License**.
