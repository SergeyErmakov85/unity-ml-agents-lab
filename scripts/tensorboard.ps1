<#
.SYNOPSIS
    Открывает TensorBoard по каталогу results/.

.PARAMETER RunId
    Показать только один запуск (results/<RunId>) вместо всех.

.PARAMETER Port
    Порт, по умолчанию 6006.

.EXAMPLE
    scripts\tensorboard.ps1
.EXAMPLE
    scripts\tensorboard.ps1 -RunId roller-01
#>
[CmdletBinding()]
param(
    [string]$RunId,
    [int]$Port = 6006
)

. "$PSScriptRoot\_common.ps1"
$ErrorActionPreference = 'Stop'

$tb = Get-VenvExe 'tensorboard'
if (-not (Test-Path $tb)) { throw "TensorBoard не найден. Выполните scripts\setup-python.ps1" }

$logdir = Join-Path (Get-RepoRoot) 'results'
if ($RunId) { $logdir = Join-Path $logdir $RunId }
if (-not (Test-Path $logdir)) { throw "Нет результатов обучения: $logdir" }

Write-Host "TensorBoard: http://localhost:$Port  (logdir = $logdir)" -ForegroundColor Green
& $tb --logdir $logdir --port $Port
