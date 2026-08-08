<#
.SYNOPSIS
    Создаёт новую среду в Assets/ML-ENVIRONMENTS из шаблона tools/env-template.

.DESCRIPTION
    Разворачивает рабочую заготовку: подкласс Agent, editor-скрипт сборки сцены,
    trainer-YAML и README. Сразу после создания среду можно собрать
    (scripts\build-scenes.ps1 <Имя>) и запустить обучение (scripts\train.ps1 <Имя>).

.PARAMETER Name
    Имя среды = имя папки и префикс классов. Только латиница, цифры и «_».

.PARAMETER Category
    Папка категории, например 04-Physics. По умолчанию 06-Custom-Environments.

.PARAMETER Behavior
    Behavior Name для Behavior Parameters и ключа в YAML.
    По умолчанию <Name>Agent. Должен быть уникален в пределах проекта.

.EXAMPLE
    scripts\new-environment.ps1 -Name Pendulum -Category 04-Physics
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)][string]$Name,
    [string]$Category = '06-Custom-Environments',
    [string]$Behavior
)

. "$PSScriptRoot\_common.ps1"
$ErrorActionPreference = 'Stop'
$root = Get-RepoRoot

if ($Name -notmatch '^[A-Za-z][A-Za-z0-9_]*$') {
    throw "Имя '$Name' недопустимо: нужны латиница/цифры/подчёркивание, первый символ — буква."
}
if (-not $Behavior) { $Behavior = "${Name}Agent" }

$categoryDir = Join-Path $root "Assets\ML-ENVIRONMENTS\$Category"
if (-not (Test-Path $categoryDir)) {
    $available = (Get-ChildItem (Join-Path $root 'Assets\ML-ENVIRONMENTS') -Directory | Where-Object Name -match '^\d\d-').Name
    throw "Категория '$Category' не найдена. Доступные: $($available -join ', ')"
}

$envDir = Join-Path $categoryDir $Name
if (Test-Path $envDir) { throw "Среда уже существует: $envDir" }

# Behavior-имена должны быть уникальны в проекте — иначе mlagents-learn перепутает агентов.
$taken = @(Get-Environments | ForEach-Object { $_.Behaviors })
if ($taken -contains $Behavior) {
    throw "Behavior Name '$Behavior' уже занят другой средой. Задайте другой через -Behavior."
}

$relRoot = "Assets/ML-ENVIRONMENTS/$Category/$Name"
$tokens = @{
    '__ENV__'       = $Name
    '__ENV_LOWER__' = $Name.ToLower()
    '__BEHAVIOR__'  = $Behavior
    '__CATEGORY__'  = $Category
    '__ROOT__'      = $relRoot
}

function Expand-Template {
    param([string]$From, [string]$To)
    $text = Get-Content -LiteralPath $From -Raw
    foreach ($k in $tokens.Keys) { $text = $text.Replace($k, $tokens[$k]) }
    New-Item -ItemType Directory -Force -Path (Split-Path $To) | Out-Null
    # UTF-8 без BOM — так Unity и git читают файлы одинаково.
    [IO.File]::WriteAllText($To, $text, (New-Object Text.UTF8Encoding $false))
    Write-Host "  + $($To.Substring($root.Length + 1))"
}

$tpl = Join-Path $root 'tools\env-template'
Write-Host "Создаю среду $Name в $Category (behavior: $Behavior)" -ForegroundColor Cyan

Expand-Template "$tpl\Scripts\__ENV__Agent.cs.txt"   (Join-Path $envDir "Scripts\${Name}Agent.cs")
Expand-Template "$tpl\Editor\__ENV__Setup.cs.txt"    (Join-Path $envDir "Editor\${Name}Setup.cs")
Expand-Template "$tpl\config\__BEHAVIOR__.yaml.txt"  (Join-Path $envDir "config\$Behavior.yaml")
Expand-Template "$tpl\README.md.txt"                 (Join-Path $envDir 'README.md')
New-Item -ItemType Directory -Force -Path (Join-Path $envDir 'Scenes') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $envDir 'Materials') | Out-Null

Write-Host ""
Write-Host "Готово: $relRoot" -ForegroundColor Green
Write-Host "Дальше:"
Write-Host "  1) открыть проект в Unity (нужна компиляция новых скриптов)"
Write-Host "  2) scripts\build-scenes.ps1 $Name        # собрать сцену"
Write-Host "  3) scripts\train.ps1 $Name -RunId $($Name.ToLower())-01"
