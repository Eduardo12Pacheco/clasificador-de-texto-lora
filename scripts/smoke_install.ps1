<#
.SYNOPSIS
    End-to-end installation smoke test for the published wheel (S19).

.DESCRIPTION
    Builds the wheel, installs it into a throw-away uv tool environment, runs
    the fixture quickstart, asserts that torch is absent and checks /healthz.
    The installation lives under the temporary directory; -Uninstall removes
    the one recorded by the previous run.

.PARAMETER Check
    Validate prerequisites only.

.PARAMETER Uninstall
    Remove the tool environment left by the previous run.

.PARAMETER Keep
    Keep the previous successful installation instead of replacing it.
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$Uninstall,
    [switch]$Keep
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$StateFile = Join-Path $env:TEMP "stc-smoke-last-$env:USERNAME.txt"
$ToolName = "spanish-text-classifier-lora"

function Test-Prerequisites {
    $missing = @()
    foreach ($command in @("uv", "python")) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            $missing += $command
        }
    }
    if ($missing.Count -gt 0) {
        throw "required command not found: $($missing -join ', ')"
    }
    $uvVersion = (& uv --version) -replace '^uv ', ''
    $pythonVersion = & python --version
    Write-Host "prerequisites OK (uv $uvVersion, $pythonVersion)"
}

function Remove-SmokeInstallation {
    if (-not (Test-Path $StateFile)) {
        Write-Host "no previous smoke installation recorded; nothing to uninstall"
        return
    }
    $root = Get-Content $StateFile -Raw
    $root = $root.Trim()
    $toolDir = Join-Path $root "tool"
    $binDir = Join-Path $root "bin"
    if (Test-Path $toolDir) {
        $env:UV_TOOL_DIR = $toolDir
        $env:UV_TOOL_BIN_DIR = $binDir
        & uv tool uninstall $ToolName 2>$null | Out-Null
    }
    Remove-Item -Recurse -Force $root -ErrorAction SilentlyContinue
    Remove-Item -Force $StateFile -ErrorAction SilentlyContinue
    Write-Host "uninstalled the smoke installation at $root"
}

if ($Check) {
    Test-Prerequisites
    exit 0
}

if ($Uninstall) {
    Remove-SmokeInstallation
    exit 0
}

Test-Prerequisites

if (-not $Keep -and (Test-Path $StateFile)) {
    $previous = (Get-Content $StateFile -Raw).Trim()
    if ($previous -like (Join-Path $env:TEMP "stc-smoke.*")) {
        Remove-Item -Recurse -Force $previous -ErrorAction SilentlyContinue
    }
}

$Root = Join-Path $env:TEMP ("stc-smoke." + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $Root | Out-Null
Set-Content -Path $StateFile -Value $Root
$ToolDir = Join-Path $Root "tool"
$BinDir = Join-Path $Root "bin"
$StcHome = Join-Path $Root "home"
$Work = Join-Path $Root "work"
foreach ($directory in @($ToolDir, $BinDir, $StcHome, $Work)) {
    New-Item -ItemType Directory -Path $directory | Out-Null
}
$env:STC_HOME = $StcHome
$env:UV_TOOL_DIR = $ToolDir
$env:UV_TOOL_BIN_DIR = $BinDir

$ServeProcess = $null
try {
    Write-Host "==> building the wheel"
    $dist = Join-Path $Work "dist"
    & uv build --wheel --out-dir $dist --no-build-logs 2>&1 | Out-Null
    $wheel = Get-ChildItem -Path $dist -Filter "*.whl" | Select-Object -First 1
    if (-not $wheel) {
        throw "uv build did not produce a wheel"
    }

    Write-Host "==> installing $($wheel.Name) into a throw-away tool environment"
    & uv tool install --python 3.12 --force $wheel.FullName 2>&1 | Out-Null
    $Stc = Join-Path $BinDir "stc.exe"
    if (-not (Test-Path $Stc)) {
        throw "stc console script was not installed"
    }

    Write-Host "==> stc --version"
    & $Stc --version
    if ($LASTEXITCODE -ne 0) { throw "stc --version failed" }

    Write-Host "==> stc doctor"
    & $Stc doctor | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "stc doctor failed" }

    Write-Host "==> stc data fixture info"
    & $Stc data fixture info
    if ($LASTEXITCODE -ne 0) { throw "stc data fixture info failed" }

    $baseline = Join-Path $Work "baseline"
    Write-Host "==> stc train baseline --dataset fixture --quick"
    & $Stc train baseline --dataset fixture --quick --out $baseline
    if ($LASTEXITCODE -ne 0) { throw "baseline training failed" }

    Write-Host "==> stc predict"
    $prediction = (& $Stc predict --model $baseline "enciende la luz" --json) | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "stc predict failed" }
    if ($prediction.schema_version -ne 1) { throw "unexpected prediction schema" }
    if ($prediction.predictions.Count -ne 1) { throw "expected exactly one prediction" }
    Write-Host ("predicted {0} ({1:N4})" -f $prediction.predictions[0].intent, $prediction.predictions[0].confidence)

    Write-Host "==> stc evaluate --split dev --dataset fixture"
    $evaluation = Join-Path $Work "eval"
    & $Stc evaluate --split dev --dataset fixture --model $baseline --out $evaluation | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "stc evaluate failed" }
    if (-not (Test-Path (Join-Path $evaluation "evaluation.json"))) {
        throw "evaluation.json was not written"
    }

    Write-Host "==> the tool environment must not contain torch"
    $ToolPython = Join-Path $ToolDir "$ToolName\Scripts\python.exe"
    if (-not (Test-Path $ToolPython)) {
        throw "could not locate the tool environment interpreter"
    }
    & $ToolPython -c "import torch" 2>$null
    if ($LASTEXITCODE -eq 0) {
        throw "torch is importable inside the installed tool environment"
    }
    Write-Host "torch is not importable (S12)"

    Write-Host "==> stc serve /healthz"
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    $listener.Stop()
    $env:STC_API_PORT = "$port"
    $env:STC_MODEL = $baseline
    $ServeProcess = Start-Process -FilePath $Stc `
        -ArgumentList @("serve", "--port", "$port") `
        -PassThru -NoNewWindow `
        -RedirectStandardOutput (Join-Path $Work "serve.log") `
        -RedirectStandardError (Join-Path $Work "serve.err.log")
    $healthy = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $payload = Invoke-RestMethod -Uri "http://127.0.0.1:$port/healthz" -TimeoutSec 2
            if ($payload.status -eq "ok") {
                $healthy = $true
                Write-Host "healthz OK: status=$($payload.status) model_loaded=$($payload.model_loaded)"
                break
            }
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $healthy) {
        if ($ServeProcess.HasExited) {
            Write-Host "serve exited early with code $($ServeProcess.ExitCode)"
        }
        foreach ($log in @("serve.log", "serve.err.log")) {
            $logPath = Join-Path $Work $log
            if (Test-Path $logPath) {
                Write-Host "--- $log (tail)"
                Get-Content -Path $logPath -Tail 40
            }
        }
        throw "serve did not answer /healthz within 60 seconds"
    }

    Write-Host "SMOKE OK"
}
catch {
    Write-Host "smoke installation failed; artifacts kept in $Root" -ForegroundColor Red
    throw
}
finally {
    if ($ServeProcess -and -not $ServeProcess.HasExited) {
        Stop-Process -Id $ServeProcess.Id -Force -ErrorAction SilentlyContinue
    }
}
