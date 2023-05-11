# CarnivoreID-App

## Prerequisites

Using `python==3.9.16`.

## Run Application
The application is orchestrated using `docker-compose` - a tool for defining and running multi-container Docker application. 

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
