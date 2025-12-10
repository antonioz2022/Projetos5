# 🐳 Script Docker - Execução com 1 Clique!

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  🐳 Dashboard de Mobilidade Urbana" -ForegroundColor Cyan
Write-Host "  Inicialização Docker" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verifica se Docker está instalado
Write-Host "Verificando Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✓ Docker encontrado: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker não encontrado!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Instale Docker Desktop de: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

Write-Host ""

# Verifica se Docker está rodando
Write-Host "Verificando se Docker está rodando..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "✓ Docker está rodando!" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker não está rodando!" -ForegroundColor Red
    Write-Host "  Inicie o Docker Desktop e tente novamente." -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

Write-Host ""

# Sobe o container (no diretório streamlit_app)
Write-Host "Iniciando dashboard..." -ForegroundColor Yellow
Write-Host ""

Push-Location "$PSScriptRoot/streamlit_app"
try {
    docker-compose up -d --build
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✓ Dashboard Iniciado com Sucesso!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Acesse em: " -NoNewline
Write-Host "http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Ver logs: " -NoNewline
Write-Host "cd streamlit_app; docker-compose logs -f" -ForegroundColor Yellow
Write-Host ""
Write-Host "🛑 Parar: " -NoNewline
Write-Host "cd streamlit_app; docker-compose down" -ForegroundColor Yellow
Write-Host ""
Write-Host "Abrindo navegador..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
Start-Process "http://localhost:8501"
