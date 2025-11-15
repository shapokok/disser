# Crop Disease Detection System - Docker Start Script (PowerShell)
# This script builds and starts the Docker containers on Windows

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Crop Disease Detection System - Docker" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not installed or not running." -ForegroundColor Red
    Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    Write-Host "Or start Docker Desktop if it's already installed." -ForegroundColor Yellow
    exit 1
}

# Check if Docker Compose is available
try {
    $composeVersion = docker-compose --version
    Write-Host "✓ Docker Compose found: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose is not available." -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host ""
Write-Host "📁 Creating necessary directories..." -ForegroundColor Yellow

$directories = @(
    "data\uploads",
    "data\test_images",
    "models",
    "results\heatmaps"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   Created: $dir" -ForegroundColor Gray
    }
}

# Check if we should use production profile
$profile = ""
if ($args[0] -eq "production" -or $args[0] -eq "prod") {
    Write-Host ""
    Write-Host "🚀 Starting in PRODUCTION mode (with Nginx)" -ForegroundColor Yellow
    $profile = "--profile production"
} else {
    Write-Host ""
    Write-Host "🔧 Starting in DEVELOPMENT mode" -ForegroundColor Green
    $profile = ""
}

# Build and start containers
Write-Host ""
Write-Host "🐳 Building Docker images..." -ForegroundColor Cyan
Write-Host "(This may take 10-15 minutes on first run...)" -ForegroundColor Gray

try {
    docker-compose build
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed"
    }
} catch {
    Write-Host "❌ Failed to build Docker images" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🚀 Starting containers..." -ForegroundColor Cyan

try {
    if ($profile) {
        docker-compose up -d $profile.Split()
    } else {
        docker-compose up -d
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Docker up failed"
    }
} catch {
    Write-Host "❌ Failed to start containers" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "⏳ Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if backend is healthy
Write-Host ""
Write-Host "🔍 Checking service health..." -ForegroundColor Cyan

$maxRetries = 30
$retryCount = 0
$backendHealthy = $false

while ($retryCount -lt $maxRetries) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/" -Method Get -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ Backend is healthy!" -ForegroundColor Green
            $backendHealthy = $true
            break
        }
    } catch {
        # Backend not ready yet
    }

    $retryCount++
    Write-Host "Waiting for backend... ($retryCount/$maxRetries)" -ForegroundColor Gray
    Start-Sleep -Seconds 2
}

if (-not $backendHealthy) {
    Write-Host "⚠ Backend took longer than expected to start" -ForegroundColor Yellow
    Write-Host "Check logs with: docker-compose logs backend" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✓ System is running!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📍 Access points:" -ForegroundColor White
Write-Host "   Backend API: http://localhost:5000" -ForegroundColor Cyan

if ($profile) {
    Write-Host "   Frontend:    http://localhost (via Nginx)" -ForegroundColor Cyan
} else {
    $frontendPath = Join-Path $PSScriptRoot "crop_monitoring_app\frontend\index.html"
    Write-Host "   Frontend:    Open $frontendPath in browser" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "📊 Useful commands:" -ForegroundColor White
Write-Host "   View logs:        docker-compose logs -f" -ForegroundColor Gray
Write-Host "   Stop:             docker-compose stop" -ForegroundColor Gray
Write-Host "   Stop & remove:    docker-compose down" -ForegroundColor Gray
Write-Host "   Restart:          docker-compose restart" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to view logs (or close this window)" -ForegroundColor Yellow
Write-Host ""

# Optionally show logs
$showLogs = Read-Host "Do you want to see container logs? (y/n)"
if ($showLogs -eq "y" -or $showLogs -eq "Y") {
    Write-Host ""
    Write-Host "Showing logs (Press Ctrl+C to exit)..." -ForegroundColor Cyan
    docker-compose logs -f
}
