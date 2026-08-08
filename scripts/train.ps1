<#
.SYNOPSIS
    Запускает mlagents-learn для среды из Assets/ML-ENVIRONMENTS.

.DESCRIPTION
    Находит trainer-YAML внутри папки среды (<Среда>/config/*.yaml), поднимает
    mlagents-learn из .venv и ждёт подключения Unity — после появления строки
    "Listening on port 5004. Start training by pressing the Play button"
    нажмите Play в редакторе.

.PARAMETER Environment
    Имя среды (папки), например Greed_world или Hit_the_ball.
    Допускается частичное совпадение. Можно также передать путь к YAML напрямую.

.PARAMETER RunId
    Идентификатор запуска. По умолчанию: <среда>-<yyyyMMdd-HHmmss>.

.PARAMETER Config
    Явный YAML, если в папке среды их несколько.

.PARAMETER Resume
    Продолжить прерванный запуск с тем же -RunId.

.PARAMETER Force
    Перезаписать результаты существующего -RunId.

.PARAMETER Inference
    Прогон обученной модели без обучения.

.PARAMETER NumEnvs
    Количество параллельных исполняемых сборок среды (требует -EnvPath).

.PARAMETER EnvPath
    Путь к собранному .exe среды. Без него обучение идёт через редактор.

.PARAMETER List
    Показать все среды, их сцены, конфиги и behavior-имена, и выйти.

.PARAMETER ExtraArgs
    Всё после `--` уходит в mlagents-learn как есть.

.EXAMPLE
    scripts\train.ps1 -List
.EXAMPLE
    scripts\train.ps1 Greed_world
.EXAMPLE
    scripts\train.ps1 Hit_the_ball -RunId roller-01
.EXAMPLE
    scripts\train.ps1 Hit_the_ball -RunId roller-01 -Resume
.EXAMPLE
    scripts\train.ps1 Hit_the_ball -- --time-scale=20 --no-graphics
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Environment,
    [string]$RunId,
    [string]$Config,
    [switch]$Resume,
    [switch]$Force,
    [switch]$Inference,
    [int]$NumEnvs = 0,
    [string]$EnvPath,
    [switch]$List,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ExtraArgs
)

. "$PSScriptRoot\_common.ps1"
$ErrorActionPreference = 'Stop'
$root = Get-RepoRoot

if ($List -or -not $Environment) {
    Write-Host "Среды в Assets/ML-ENVIRONMENTS:" -ForegroundColor Cyan
    foreach ($e in Get-Environments) {
        Write-Host ""
        Write-Host ("  {0}  [{1}]" -f $e.Name, $e.Category) -ForegroundColor White
        if ($e.Scenes.Count) {
            Write-Host ("    сцены:     " + (($e.Scenes | ForEach-Object { $_.Name }) -join ', '))
        }
        else { Write-Host "    сцены:     — (нет .unity)" -ForegroundColor DarkYellow }
        if ($e.Configs.Count) {
            Write-Host ("    конфиги:   " + (($e.Configs | ForEach-Object { $_.Name }) -join ', '))
            Write-Host ("    behaviors: " + ($e.Behaviors -join ', '))
        }
        else { Write-Host "    конфиги:   — (нет config/*.yaml, обучение не запустится)" -ForegroundColor DarkYellow }
    }
    Write-Host ""
    if (-not $Environment) { Write-Host "Запуск: scripts\train.ps1 <Среда> [-RunId <id>]" }
    return
}

$learn = Assert-Venv

# --- выбор конфига ---------------------------------------------------------
$envName = $null
$configPath = $null
if ($Config) {
    $configPath = (Resolve-Path -LiteralPath $Config).Path
}
elseif ((Test-Path -LiteralPath $Environment) -and ($Environment -match '\.ya?ml$')) {
    $configPath = (Resolve-Path -LiteralPath $Environment).Path
    $envName = [IO.Path]::GetFileNameWithoutExtension($configPath)
}
else {
    $env_ = Resolve-Environment $Environment
    $envName = $env_.Name
    if ($env_.Configs.Count -eq 0) {
        throw "У среды '$($env_.Name)' нет config/*.yaml. Создайте его (см. Assets/ML-ENVIRONMENTS/_TEMPLATE/config) или укажите -Config."
    }
    if ($env_.Configs.Count -gt 1) {
        throw "У среды '$($env_.Name)' несколько конфигов: $(($env_.Configs.Name) -join ', '). Укажите нужный через -Config."
    }
    $configPath = $env_.Configs[0].FullName
    Write-Host "Среда:   $($env_.Name) [$($env_.Category)]" -ForegroundColor Cyan
    if ($env_.Scenes.Count) {
        $rel = $env_.Scenes[0].FullName.Substring($root.Length + 1)
        Write-Host "Сцена:   $rel" -ForegroundColor Cyan
    }
}
if (-not $envName) { $envName = 'run' }

$behaviors = @(Get-BehaviorNames $configPath)
Write-Host "Конфиг:  $($configPath.Substring($root.Length + 1))" -ForegroundColor Cyan
Write-Host "Behavior: $($behaviors -join ', ')" -ForegroundColor Cyan

if (-not $RunId) {
    $RunId = "{0}-{1}" -f $envName.ToLower(), (Get-Date -Format 'yyyyMMdd-HHmmss')
}

# --- сборка аргументов -----------------------------------------------------
$argv = @($configPath, "--run-id=$RunId", "--results-dir=$(Join-Path $root 'results')")
if ($Resume) { $argv += '--resume' }
if ($Force) { $argv += '--force' }
if ($Inference) { $argv += '--inference' }
if ($EnvPath) { $argv += "--env=$EnvPath" }
if ($NumEnvs -gt 0) {
    if (-not $EnvPath) { throw '-NumEnvs требует -EnvPath (собранный билд среды).' }
    $argv += "--num-envs=$NumEnvs"
}
if ($ExtraArgs) { $argv += ($ExtraArgs | Where-Object { $_ -ne '--' }) }

Write-Host "Run id:  $RunId" -ForegroundColor Cyan
Write-Host ""
if (-not $EnvPath) {
    Write-Host "Когда появится 'Listening on port 5004' — нажмите Play в Unity." -ForegroundColor Yellow
    Write-Host ""
}

Push-Location $root
try {
    & $learn @argv
    $code = $LASTEXITCODE
}
finally { Pop-Location }

if ($code -eq 0) {
    Write-Host ""
    Write-Host "Готово. Результаты: results\$RunId" -ForegroundColor Green
    Write-Host "Метрики:  scripts\tensorboard.ps1"
    Write-Host "Модель:  results\$RunId\*.onnx  — перетащите в Behavior Parameters > Model"
}
exit $code
