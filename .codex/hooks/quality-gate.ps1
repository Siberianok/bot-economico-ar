[CmdletBinding()]
param(
    [ValidateSet("Informational", "Blocking")]
    [string]$Mode = "Informational"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$expectedProjectRoot = [System.IO.Path]::GetFullPath("C:\Users\ariel\Downloads\IA\ObeconBot")
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$frontendRoot = Join-Path $projectRoot "frontend"
$tempRoot = Join-Path $projectRoot ".codex\.tmp"
$runId = [System.Guid]::NewGuid().ToString("N")
$runRoot = Join-Path $tempRoot ("quality-gate-{0}" -f $runId)
$failures = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]
$passed = 0

function Invoke-GateStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action,
        [switch]$WarningOnly
    )

    Write-Host ("[RUN ] {0}" -f $Name)
    try {
        & $Action
        $script:passed += 1
        Write-Host ("[ OK ] {0}" -f $Name) -ForegroundColor Green
    }
    catch {
        $message = "{0}: {1}" -f $Name, $_.Exception.Message
        if ($WarningOnly) {
            $script:warnings.Add($message)
            Write-Host ("[WARN] {0}" -f $message) -ForegroundColor Yellow
        }
        else {
            $script:failures.Add($message)
            Write-Host ("[FAIL] {0}" -f $message) -ForegroundColor Red
        }
    }
}

function Assert-NativeSuccess {
    param([string]$CommandName)
    if ($LASTEXITCODE -ne 0) {
        throw ("{0} returned exit code {1}" -f $CommandName, $LASTEXITCODE)
    }
}

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

$oldTelegramToken = $env:TELEGRAM_TOKEN
$oldBotToken = $env:BOT_TOKEN
$oldDatabaseUrl = $env:DATABASE_URL
$oldLegacyFetch = $env:LEGACY_LIVE_FETCH_ENABLED
$oldBytecode = $env:PYTHONDONTWRITEBYTECODE

try {
    Push-Location $projectRoot

    $env:TELEGRAM_TOKEN = "quality-gate-placeholder-token"
    $env:BOT_TOKEN = "quality-gate-placeholder-token"
    $env:DATABASE_URL = "sqlite:///./.codex/.tmp/quality-gate-$runId/quality_gate.db"
    $env:LEGACY_LIVE_FETCH_ENABLED = "0"
    $env:PYTHONDONTWRITEBYTECODE = "1"

    Invoke-GateStep -Name "Project boundary" -Action {
        if (-not [string]::Equals($projectRoot, $expectedProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw ("Unexpected project root: {0}" -f $projectRoot)
        }
        if (-not (Test-Path (Join-Path $projectRoot "bot_econ_full_plus_rank_alerts.py"))) {
            throw "Legacy bot entrypoint is missing"
        }
    }

    Invoke-GateStep -Name "Auditor configuration" -Action {
        $requiredFiles = @(
            "AGENTS.md",
            ".codex\hooks.json",
            ".codex\agents\auditor_obeconbot.toml",
            ".codex\hooks\quality-gate.ps1"
        )
        foreach ($relativePath in $requiredFiles) {
            if (-not (Test-Path (Join-Path $projectRoot $relativePath))) {
                throw ("Required auditor file is missing: {0}" -f $relativePath)
            }
        }
        Get-Content -Raw (Join-Path $projectRoot ".codex\hooks.json") | ConvertFrom-Json | Out-Null
    }

    Invoke-GateStep -Name "Python runtime" -Action {
        if (-not (Test-Path $python)) {
            throw ("Project Python was not found: {0}" -f $python)
        }
        & $python --version
        Assert-NativeSuccess "python --version"
    }

    Invoke-GateStep -Name "Backend import" -Action {
        & $python -c "import backend; import backend.app.main; assert backend.app.main.app.title == 'ObeconBot API'; print('backend import OK')"
        Assert-NativeSuccess "backend import"
    }

    Invoke-GateStep -Name "Full regression tests" -Action {
        $pytestBaseTemp = Join-Path $runRoot "pytest"
        & $python -m pytest tests -q -p no:cacheprovider --tb=short --basetemp $pytestBaseTemp
        Assert-NativeSuccess "full pytest"
    }

    Invoke-GateStep -Name "Python dependency consistency" -Action {
        & $python -m pip check
        Assert-NativeSuccess "pip check"
    }

    Invoke-GateStep -Name "V2 Python lint" -Action {
        & $python -m ruff check backend bot main.py metrics.py tests --ignore E402 --output-format concise --no-cache
        Assert-NativeSuccess "ruff check"
    }

    Invoke-GateStep -Name "Cross-project contamination markers" -Action {
        $patterns = @(
            "ArgentinaStrategy",
            "evaluate_argentina",
            "/api/argentina",
            "MarketReferenceSnapshot",
            "OpportunityObservation",
            "local_usdt_ars_cross_exchange",
            "crypto_vs_official_spread"
        )
        $scanRoots = @("backend", "bot", "tests", "docs", "frontend\src")
        $files = foreach ($scanRoot in $scanRoots) {
            $path = Join-Path $projectRoot $scanRoot
            if (Test-Path $path) {
                Get-ChildItem -Path $path -Recurse -File -ErrorAction Stop |
                    Where-Object { $_.Extension -in @(".py", ".ts", ".tsx", ".md", ".json", ".yaml", ".yml") }
            }
        }
        $matches = $files | Select-String -SimpleMatch -Pattern $patterns
        if ($matches) {
            throw ("Unexpected contamination markers found: {0}" -f (($matches | Select-Object -ExpandProperty Path -Unique) -join ", "))
        }
    }

    Invoke-GateStep -Name "Frontend production build" -WarningOnly:($Mode -eq "Informational") -Action {
        if (-not (Test-Path (Join-Path $frontendRoot "package.json"))) {
            throw "frontend/package.json is missing"
        }
        $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if (-not $npmCommand) {
            throw "npm.cmd is not available"
        }
        Push-Location $frontendRoot
        try {
            $buildSucceeded = $false
            foreach ($attempt in 1..3) {
                & $npmCommand.Source run build
                if ($LASTEXITCODE -eq 0) {
                    $buildSucceeded = $true
                    break
                }
                if ($attempt -lt 3) {
                    Write-Host ("[INFO] Frontend build attempt {0} failed; retrying after a short delay." -f $attempt) -ForegroundColor Yellow
                    Start-Sleep -Seconds 2
                }
            }
            if (-not $buildSucceeded) {
                throw "npm run build failed after 3 attempts"
            }
        }
        finally {
            Pop-Location
        }
    }

    Invoke-GateStep -Name "Frontend strict typecheck" -WarningOnly -Action {
        $package = Get-Content -Raw (Join-Path $frontendRoot "package.json") | ConvertFrom-Json
        $devNames = @($package.devDependencies.PSObject.Properties.Name)
        if ($devNames -notcontains "@types/react" -or $devNames -notcontains "@types/react-dom") {
            throw "Known baseline debt: @types/react and @types/react-dom are not declared"
        }
        $tsc = Join-Path $frontendRoot "node_modules\.bin\tsc.cmd"
        if (-not (Test-Path $tsc)) {
            throw "Local TypeScript compiler is unavailable"
        }
        Push-Location $frontendRoot
        try {
            & $tsc --noEmit
            Assert-NativeSuccess "tsc --noEmit"
        }
        finally {
            Pop-Location
        }
    }
}
finally {
    if ((Get-Location).Path -ne $projectRoot) {
        Set-Location $projectRoot
    }
    Pop-Location

    $env:TELEGRAM_TOKEN = $oldTelegramToken
    $env:BOT_TOKEN = $oldBotToken
    $env:DATABASE_URL = $oldDatabaseUrl
    $env:LEGACY_LIVE_FETCH_ENABLED = $oldLegacyFetch
    $env:PYTHONDONTWRITEBYTECODE = $oldBytecode

    if (Test-Path $runRoot) {
        Remove-Item -LiteralPath $runRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ""
Write-Host "ObeconBot quality gate summary"
Write-Host ("Mode: {0}" -f $Mode)
Write-Host ("Passed steps: {0}" -f $passed)
Write-Host ("Failures: {0}" -f $failures.Count)
Write-Host ("Warnings: {0}" -f $warnings.Count)

foreach ($warning in $warnings) {
    Write-Host ("[WARN] {0}" -f $warning) -ForegroundColor Yellow
}
foreach ($failure in $failures) {
    Write-Host ("[FAIL] {0}" -f $failure) -ForegroundColor Red
}

if ($failures.Count -gt 0 -and $Mode -eq "Blocking") {
    exit 1
}

if ($failures.Count -gt 0) {
    Write-Host "Informational mode: failures were reported without blocking completion." -ForegroundColor Yellow
}

exit 0
