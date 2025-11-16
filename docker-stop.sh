#!/bin/bash

# Crop Disease Detection System - Docker Stop Script

set -e

echo "=========================================="
echo "Stopping Crop Disease Detection System"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Check if containers are running
if ! docker-compose ps | grep -q "Up"; then
    echo "ℹ️  No containers are currently running"
    exit 0
fi

# Ask for confirmation if removing volumes
if [ "$1" == "--clean" ] || [ "$1" == "-c" ]; then
    echo -e "${RED}⚠️  This will remove containers AND volumes (uploaded images, results)${NC}"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi

    echo ""
    echo "🗑️  Stopping and removing containers and volumes..."
    docker-compose down -v
    echo -e "${GREEN}✓ Containers and volumes removed${NC}"
else
    echo "🛑 Stopping containers..."
    docker-compose stop

    echo ""
    echo -e "${GREEN}✓ Containers stopped${NC}"
    echo ""
    echo "💡 To completely remove containers: docker-compose down"
    echo "💡 To remove containers AND data: ./docker-stop.sh --clean"
fi

echo ""
