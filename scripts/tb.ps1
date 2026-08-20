# Запуск TensorBoard по всем прогонам лаборатории (требование 11.6).
#
# Использование:
#   .\scripts\tb.ps1                 # порт 6006, все примеры
#   .\scripts\tb.ps1 -Port 6007      # другой порт
#   .\scripts\tb.ps1 -EnvId E03_RollerBall   # только одна среда
#
# Интерпретатор берётся из python\.venv — TensorBoard не нужно ставить глобально.

[CmdletBinding()]
param(
    [int]$Port = 6006,
    [string]$EnvId = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "python\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Не найдено окружение $python. Создайте его: py -3.10 -m venv python\.venv"
}

$logdir = Join-Path $repoRoot "results"
if ($EnvId -ne "") {
    $logdir = Join-Path $logdir $EnvId
}

if (-not (Test-Path $logdir)) {
    Write-Error "Каталог прогонов не найден: $logdir. Сначала запустите обучение."
}

Write-Host "TensorBoard: $logdir  ->  http://localhost:$Port"
& $python -m tensorboard.main --logdir $logdir --port $Port
