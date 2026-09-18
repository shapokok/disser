# Quick script to rebuild Docker with export functionality

Write-Host "Stopping current container..." -ForegroundColor Yellow
docker-compose down

Write-Host "`nRebuilding Docker image with export functionality..." -ForegroundColor Cyan
docker-compose build --no-cache

Write-Host "`nStarting container..." -ForegroundColor Green
docker-compose up -d

Write-Host "`nWaiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "`nChecking backend health..." -ForegroundColor Cyan
$maxRetries = 20
$retryCount = 0

while ($retryCount -lt $maxRetries) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5000/" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "Backend is ready!" -ForegroundColor Green
            break
        }
    } catch {
        # Not ready yet
    }
    $retryCount++
    Write-Host "Waiting... ($retryCount/$maxRetries)"
    Start-Sleep -Seconds 2
}

Write-Host "`nVerifying export functionality..." -ForegroundColor Cyan
docker exec crop-disease-backend python -c "from export_utils import export_to_excel; print('Export modules loaded successfully!')"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Done! Backend is running on http://localhost:5000" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nOpen crop_monitoring_app\frontend\analyze.html to test export!" -ForegroundColor Yellow
