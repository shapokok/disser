# Crop Disease Detection System - Docker Stop Script (PowerShell)
# This script stops the Docker containers

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Stopping Crop Disease Detection System" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker ps | Out-Null
} catch {
    Write-Host "❌ Docker is not running." -ForegroundColor Red
    exit 1
}

# Ask user what they want to do
Write-Host "Choose an option:" -ForegroundColor Yellow
Write-Host "  1) Stop containers (keep data)" -ForegroundColor White
Write-Host "  2) Stop and remove containers (keep data)" -ForegroundColor White
Write-Host "  3) Stop and remove everything including volumes (⚠ deletes data)" -ForegroundColor Red
Write-Host ""

$choice = Read-Host "Enter choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "⏸ Stopping containers..." -ForegroundColor Yellow
        docker-compose stop
        Write-Host "✓ Containers stopped" -ForegroundColor Green
        Write-Host "To restart: docker-compose start" -ForegroundColor Gray
    }
    "2" {
        Write-Host ""
        Write-Host "🗑 Stopping and removing containers..." -ForegroundColor Yellow
        docker-compose down
        Write-Host "✓ Containers removed" -ForegroundColor Green
        Write-Host "To restart: docker-compose up -d" -ForegroundColor Gray
    }
    "3" {
        Write-Host ""
        Write-Host "⚠ This will delete all uploaded images and results!" -ForegroundColor Red
        $confirm = Read-Host "Are you sure? (yes/no)"

        if ($confirm -eq "yes") {
            Write-Host ""
            Write-Host "🗑 Removing everything..." -ForegroundColor Red
            docker-compose down -v
            Write-Host "✓ Everything removed" -ForegroundColor Green
        } else {
            Write-Host "Cancelled." -ForegroundColor Yellow
        }
    }
    default {
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Cyan
