# 🛑 Script para Parar o Dashboard Docker

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Parando Dashboard..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

docker-compose down

Write-Host ""
Write-Host "✓ Dashboard parado com sucesso!" -ForegroundColor Green
Write-Host ""
