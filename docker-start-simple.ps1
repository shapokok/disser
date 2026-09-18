# Simple Docker Start Script
Write-Host "Starting Crop Disease Detection System..." -ForegroundColor Cyan

# Create directories
New-Item -ItemType Directory -Path "data\uploads" -Force -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "models" -Force -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "results\heatmaps" -Force -ErrorAction SilentlyContinue | Out-Null

# Start Docker
docker-compose up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS! System is starting..." -ForegroundColor Green
    Write-Host ""
    Write-Host "Access points:" -ForegroundColor White
    Write-Host "  Backend API: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "  Frontend:    Open frontend\index.html in browser" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "View logs:  docker-compose logs -f" -ForegroundColor Gray
    Write-Host "Stop:       docker-compose down" -ForegroundColor Gray
} else {
    Write-Host "ERROR: Failed to start Docker" -ForegroundColor Red
}
