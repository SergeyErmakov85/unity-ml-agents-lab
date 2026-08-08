<#
.SYNOPSIS
    Создаёт .venv с Python 3.10.12 и ставит mlagents для обучения агентов.

.DESCRIPTION
    Использует uv (https://docs.astral.sh/uv/). Если uv не установлен — скрипт
    подскажет команду установки. Версии зафиксированы в requirements.txt,
    точный снимок рабочего окружения — в requirements.lock.txt.

.PARAMETER Locked
    Ставить точные версии из requirements.lock.txt вместо requirements.txt.

.PARAMETER Cuda
    Собрать torch с поддержкой CUDA, например -Cuda cu121 или -Cuda cu124.
    По умолчанию ставится CPU-сборка с PyPI.

.PARAMETER Recreate
    Удалить существующий .venv и создать заново.

.EXAMPLE
    scripts\setup-python.ps1
.EXAMPLE
    scripts\setup-python.ps1 -Cuda cu121
.EXAMPLE
    scripts\setup-python.ps1 -Locked -Recreate
#>
[CmdletBinding()]
param(
    [switch]$Locked,
    [string]$Cuda,
    [switch]$Recreate
)

. "$PSScriptRoot\_common.ps1"
$ErrorActionPreference = 'Stop'

$root = Get-RepoRoot
$venv = Get-VenvPath
$pyVersion = (Get-Content -LiteralPath (Join-Path $root '.python-version') -TotalCount 1).Trim()

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv не найден. Установите его одной из команд:" -ForegroundColor Yellow
    Write-Host '  winget install --id=astral-sh.uv -e'
    Write-Host '  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"'
    throw 'uv is required'
}

if ($Recreate -and (Test-Path $venv)) {
    Write-Host "Удаляю существующий $venv ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $venv
}

Write-Host "==> Python $pyVersion" -ForegroundColor Cyan
uv python install $pyVersion
if ($LASTEXITCODE -ne 0) { throw "uv python install $pyVersion failed" }

if (-not (Test-Path $venv)) {
    Write-Host "==> Создаю виртуальное окружение $venv" -ForegroundColor Cyan
    uv venv --python $pyVersion $venv
    if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
}

if ($Cuda) {
    $idx = "https://download.pytorch.org/whl/$Cuda"
    Write-Host "==> torch (CUDA $Cuda) из $idx" -ForegroundColor Cyan
    uv pip install --python $venv 'torch~=2.2.1' --index-url $idx
    if ($LASTEXITCODE -ne 0) { throw 'torch install failed' }
}

$req = if ($Locked) { 'requirements.lock.txt' } else { 'requirements.txt' }
Write-Host "==> Ставлю зависимости из $req" -ForegroundColor Cyan
uv pip install --python $venv -r (Join-Path $root $req)
if ($LASTEXITCODE -ne 0) { throw "uv pip install -r $req failed" }

Write-Host "==> Проверка" -ForegroundColor Cyan
$learn = Get-VenvExe 'mlagents-learn'
& $learn --help *>&1 | Select-Object -First 1 | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'mlagents-learn не запускается — смотрите вывод выше' }

$torchInfo = & (Get-VenvExe 'python') -c "import torch, mlagents.trainers as t; print(f'mlagents {t.__version__} | torch {torch.__version__} | cuda {torch.cuda.is_available()}')" 2>$null
Write-Host ""
Write-Host "Готово. $torchInfo" -ForegroundColor Green
Write-Host "Запуск обучения:  scripts\train.ps1 <Среда> -RunId <run-id>"
Write-Host "Список сред:      scripts\train.ps1 -List"
