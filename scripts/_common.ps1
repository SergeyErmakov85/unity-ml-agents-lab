# Общие функции для скриптов лаборатории. Подключается через: . "$PSScriptRoot\_common.ps1"

Set-StrictMode -Version Latest

$script:RepoRoot = Split-Path -Parent $PSScriptRoot

function Get-RepoRoot { $script:RepoRoot }

function Get-VenvPath { Join-Path (Get-RepoRoot) '.venv' }

function Get-VenvExe {
    param([Parameter(Mandatory)][string]$Name)
    Join-Path (Get-VenvPath) "Scripts\$Name.exe"
}

function Assert-Venv {
    $exe = Get-VenvExe 'mlagents-learn'
    if (-not (Test-Path $exe)) {
        throw "Python-окружение не найдено ($exe). Сначала выполните: scripts\setup-python.ps1"
    }
    $exe
}

function Get-UnityVersion {
    $f = Join-Path (Get-RepoRoot) 'ProjectSettings\ProjectVersion.txt'
    if (-not (Test-Path $f)) { return $null }
    $line = Select-String -LiteralPath $f -Pattern '^m_EditorVersion:\s*(\S+)' | Select-Object -First 1
    if ($line) { $line.Matches[0].Groups[1].Value } else { $null }
}

function Get-UnityExe {
    if ($env:UNITY_EXE -and (Test-Path $env:UNITY_EXE)) { return $env:UNITY_EXE }
    $ver = Get-UnityVersion
    $candidates = @()
    if ($ver) {
        $candidates += "C:\Program Files\Unity\Hub\Editor\$ver\Editor\Unity.exe"
        $candidates += "$env:LOCALAPPDATA\Unity\Hub\Editor\$ver\Editor\Unity.exe"
    }
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    throw "Unity $ver не найден. Укажите путь через переменную окружения UNITY_EXE."
}

# Все среды: Assets/ML-ENVIRONMENTS/<NN-Category>/<EnvName>/
function Get-Environments {
    $root = Join-Path (Get-RepoRoot) 'Assets\ML-ENVIRONMENTS'
    Get-ChildItem -Path $root -Directory | Where-Object { $_.Name -match '^\d\d-' } | ForEach-Object {
        $category = $_.Name
        Get-ChildItem -Path $_.FullName -Directory | Where-Object { $_.Name -notmatch '^[._]' } | ForEach-Object {
            $configs = @()
            $cfgDir = Join-Path $_.FullName 'config'
            if (Test-Path $cfgDir) {
                $configs = @(Get-ChildItem -Path $cfgDir -File -Include *.yaml, *.yml -Recurse)
            }
            $scenes = @()
            $sceneDir = Join-Path $_.FullName 'Scenes'
            if (Test-Path $sceneDir) {
                $scenes = @(Get-ChildItem -Path $sceneDir -File -Filter *.unity)
            }
            [pscustomobject]@{
                Name      = $_.Name
                Category  = $category
                Path      = $_.FullName
                Configs   = $configs
                Scenes    = $scenes
                Behaviors = @($configs | ForEach-Object { Get-BehaviorNames $_.FullName } | Select-Object -Unique)
            }
        }
    }
}

# Имена behavior-ов из trainer-YAML (ключи первого уровня под `behaviors:`).
function Get-BehaviorNames {
    param([Parameter(Mandatory)][string]$YamlPath)
    $names = @()
    $inBehaviors = $false
    foreach ($line in (Get-Content -LiteralPath $YamlPath)) {
        if ($line -match '^\s*#') { continue }
        if ($line -match '^behaviors:\s*$') { $inBehaviors = $true; continue }
        if (-not $inBehaviors) { continue }
        if ($line -match '^\S') { $inBehaviors = $false; continue }   # вышли из блока
        if ($line -match '^\s{2}([A-Za-z0-9_\-\.]+):\s*$') { $names += $Matches[1] }
    }
    $names
}

# Ищет среду по имени (регистронезависимо, допускается частичное совпадение).
function Resolve-Environment {
    param([Parameter(Mandatory)][string]$Name)
    $all = @(Get-Environments)
    $exact = @($all | Where-Object { $_.Name -ieq $Name })
    if ($exact.Count -eq 1) { return $exact[0] }
    $partial = @($all | Where-Object { $_.Name -ilike "*$Name*" })
    if ($partial.Count -eq 1) { return $partial[0] }
    if ($partial.Count -gt 1) {
        throw "Имя '$Name' неоднозначно: $(($partial.Name) -join ', ')"
    }
    throw "Среда '$Name' не найдена. Доступные: $(($all.Name) -join ', ')"
}
