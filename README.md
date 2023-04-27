# CarnivoreID-Web


## Development

### In Docker

Start:
```bash
docker-compose --env-file .env up --build
```

Stop:
```bash
docker-compose down
```

### On local machine

```bash
conda create -n cidapp -c conda-forge torchvision wandb django-environ django-allauth django pip pandas loguru
pip install django_q
```

```bash
cd resources
pip install fgvc-1.3.3.dev0-py3-none-any.whl
```
