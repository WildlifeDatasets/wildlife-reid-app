# CarnivoreID-App

## Run Application
The application is orchestrated using `docker-compose` - a tool for defining and running multi-container Docker application.

Before starting the application create `.env` file with secret variables.
```bash
echo "WANDB_API_KEY=..." >> .env
```

### Development
Run the following commands to build and start the application in the development mode.
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml build
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

Run the following commands to view the final development mode configuration with overrides from `docker-compose.dev.yml`. 
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml config
```
