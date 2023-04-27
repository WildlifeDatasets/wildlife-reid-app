# CarnivoreID-Web


## Development

### In Docker

Start:
```bash
docker-compose --verbose --env-file .env up --build
```

Stop:
```bash
docker-compose down
```

### On local machine

```bash
conda create -n cidapp -c conda-forge torchvision wandb django-environ django-allauth django pip pandas loguru joblib
pip install django_q
```

```bash
pip install docker/resources/fgvc-1.3.3.dev0-py3-none-any.whl
```
