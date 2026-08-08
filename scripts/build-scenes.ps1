<#
.SYNOPSIS
    Пересобирает сцены сред в batch-режиме Unity и валидирует проект.

.DESCRIPTION
    Сцены в этом репозитории генерируются кодом (Assets/.../Editor/<Name>Setup.cs,
    меню Tools/RL/...). Скрипт находит все такие методы, вызывает их через
    -executeMethod и в конце прогоняет ProjectBootstrap.ConfigureAndValidate.

.PARAMETER Environment
    Пересобрать только указанную среду.

.PARAMETER ValidateOnly
    Только ProjectBootstrap.ConfigureAndValidate, без пересборки сцен.

.PARAMETER List
    Показать найденные методы сборки и выйти.

.EXAMPLE
    scripts\build-scenes.ps1 -List
.EXAMPLE
    scripts\build-scenes.ps1
.EXAMPLE
    scripts\build-scenes.ps1 Greed_world
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Environment,
    [switch]$ValidateOnly,
    [switch]$List
)

. "$PSScriptRoot\_common.ps1"
$ErrorActionPreference = 'Stop'
$root = Get-RepoRoot

# Находит статические методы, помеченные [MenuItem("Tools/RL/...")].
function Get-BuildMethods {
    param([string]$Scope)
    $searchRoot = if ($Scope) { (Resolve-Environment $Scope).Path } else { Join-Path $root 'Assets' }
    Get-ChildItem -Path $searchRoot -Recurse -File -Filter *.cs |
        Where-Object { $_.DirectoryName -match '\\Editor(\\|$)' } |
        ForEach-Object {
            $text = Get-Content -LiteralPath $_.FullName -Raw
            $cls = [regex]::Match($text, '(?m)^\s*(?:public\s+|internal\s+)?(?:static\s+)?class\s+(\w+)')
            if (-not $cls.Success) { return }
            foreach ($m in [regex]::Matches($text,
                    '\[MenuItem\("Tools/RL/(?<menu>[^"]+)"\)\][\s\S]{0,200}?static\s+void\s+(?<method>\w+)\s*\(')) {
                [pscustomobject]@{
                    Class  = $cls.Groups[1].Value
                    Method = $m.Groups['method'].Value
                    Menu   = $m.Groups['menu'].Value
                    File   = $_.FullName.Substring($root.Length + 1)
                }
            }
        }
}

# Соглашение проекта: сборщик сцены — это пункт меню "Tools/RL/Build ...".
# Прочие пункты Tools/RL (темы, скриншоты) в batch-прогон не попадают.
$methods = @(Get-BuildMethods $Environment |
        Where-Object { $_.Class -ne 'ProjectBootstrap' -and $_.Menu -like 'Build *' } |
        Sort-Object Class, Method -Unique)

if ($List) {
    $methods | Format-Table Class, Method, Menu, File -AutoSize
    return
}

$unity = Get-UnityExe

# Unity в batch-режиме не может открыть проект, уже открытый в редакторе:
# процесс мгновенно завершается с кодом 1 и пустым логом.
if (Test-Path (Join-Path $root 'Temp\UnityLockfile')) {
    Write-Host "Проект открыт в редакторе Unity — batch-режим не сможет получить блокировку." -ForegroundColor Yellow
    Write-Host "Закройте редактор и повторите, либо выполните то же самое из меню:" -ForegroundColor Yellow
    Write-Host "  Tools → RL → Build ... / Configure Project / Validate Training Setup"
    exit 2
}

$logDir = Join-Path $root 'Logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-UnityMethod {
    param([string]$Target)
    $log = Join-Path $logDir ("build-" + ($Target -replace '\W', '_') + ".log")
    Write-Host "==> $Target" -ForegroundColor Cyan
    & $unity -batchmode -quit -projectPath $root -executeMethod $Target -logFile $log | Out-Null
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Host "    ОШИБКА (exit $code). Лог: $log" -ForegroundColor Red
        Get-Content -LiteralPath $log -Tail 40 | ForEach-Object { "      $_" }
    }
    else {
        Write-Host "    ок  ($log)" -ForegroundColor Green
    }
    $code
}

$failed = 0
if (-not $ValidateOnly) {
    if (-not $methods) { Write-Host "Методы сборки сцен не найдены." -ForegroundColor Yellow }
    foreach ($m in $methods) {
        if ((Invoke-UnityMethod "$($m.Class).$($m.Method)") -ne 0) { $failed++ }
    }
}
if ((Invoke-UnityMethod 'ProjectBootstrap.ConfigureAndValidate') -ne 0) { $failed++ }

if ($failed -gt 0) { Write-Host "`nЗавершено с ошибками: $failed" -ForegroundColor Red; exit 1 }
Write-Host "`nВсё собрано и провалидировано." -ForegroundColor Green
