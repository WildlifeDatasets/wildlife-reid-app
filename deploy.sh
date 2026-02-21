#!/bin/bash
set -e

ENV=${1:-dev}

echo "=== Deploy started at: $(date '+%Y-%m-%d %H:%M:%S') ==="

case "$ENV" in
  dev)
    COMPOSE_FILE="docker-compose.dev.yml"
    ;;
  prod)
    COMPOSE_FILE="docker-compose.yml"
    ;;
  *)
    echo "Usage: $0 {dev|prod}"
    exit 1
    ;;
esac

echo "Deploying $ENV..."
docker compose -f "$COMPOSE_FILE" down
git pull
docker compose -f "$COMPOSE_FILE" up -d --build
echo "=== Deploy finished at: $(date '+%Y-%m-%d %H:%M:%S') ==="
