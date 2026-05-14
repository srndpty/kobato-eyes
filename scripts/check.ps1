param(
    [switch]$NoCoverage,
    [switch]$Fix,
    [string]$CoverageFile = "tmp/coverage/.coverage"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

$env:PYTHONPATH = "src"
$env:KOE_HEADLESS = "1"

$ChangedPythonFiles = @(
    git diff --name-only --diff-filter=ACMRTUXB
    git diff --cached --name-only --diff-filter=ACMRTUXB
    git ls-files --others --exclude-standard
) | Where-Object { $_ -match '\.py$' } | Sort-Object -Unique

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

if ($Fix) {
    if ($ChangedPythonFiles.Count -eq 0) {
        Write-Host ""
        Write-Host "==> no changed Python files to fix" -ForegroundColor Cyan
    } else {
        Invoke-Step "isort" {
            & $Python -m isort @ChangedPythonFiles
        }

        Invoke-Step "ruff check --fix" {
            & $Python -m ruff check @ChangedPythonFiles --fix
        }

        Invoke-Step "ruff format" {
            & $Python -m ruff format @ChangedPythonFiles
        }
    }
}

Invoke-Step "git diff --check" {
    git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check
}

if ($ChangedPythonFiles.Count -eq 0) {
    Write-Host ""
    Write-Host "==> no changed Python files to check with isort/ruff format" -ForegroundColor Cyan
} else {
    Invoke-Step "isort check" {
        & $Python -m isort @ChangedPythonFiles --check-only
    }
}

Invoke-Step "ruff check" {
    & $Python -m ruff check .
}

if ($ChangedPythonFiles.Count -ne 0) {
    Invoke-Step "ruff format check" {
        & $Python -m ruff format @ChangedPythonFiles --check
    }
}

if ($NoCoverage) {
    Invoke-Step "pytest" {
        & $Python -m pytest
    }
} else {
    $CoverageDir = Split-Path $CoverageFile -Parent
    if ($CoverageDir -and -not (Test-Path -LiteralPath $CoverageDir)) {
        New-Item -ItemType Directory -Path $CoverageDir | Out-Null
    }

    $env:COVERAGE_FILE = $CoverageFile

    Invoke-Step "coverage run pytest" {
        & $Python -m coverage run -m pytest
    }

    Invoke-Step "coverage report" {
        & $Python -m coverage report --show-missing --skip-covered
    }
}
