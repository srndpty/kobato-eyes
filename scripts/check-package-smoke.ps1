param(
    [switch]$CheckWorkingTreeArtifacts
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

Write-Host "==> package tree health" -ForegroundColor Cyan
$RequiredPaths = @(
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "src",
    "tests"
)

foreach ($RequiredPath in $RequiredPaths) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required package path is missing: $RequiredPath"
    }
}

$ForbiddenPatterns = @(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    ".pytest_cache",
    ".coverage"
)

$IgnoredArtifactRoots = @(
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "tmp"
)

$ForbiddenPathPatterns = @(
    "*/__pycache__/*",
    "*/.egg-info/*",
    "*/.pytest_cache/*"
)

function Test-GeneratedArtifactPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathText
    )

    $Normalized = $PathText -replace "\\", "/"
    $Name = Split-Path -Leaf $Normalized
    foreach ($IgnoredRoot in $IgnoredArtifactRoots) {
        if ($Normalized -eq $IgnoredRoot -or $Normalized -like "$IgnoredRoot/*") {
            return $false
        }
    }
    foreach ($Pattern in $ForbiddenPathPatterns) {
        if ($Normalized -like $Pattern) {
            return $true
        }
    }
    return (
        $Name -like "*.pyc" -or
        $Name -like "*.pyo" -or
        $Name -like "*.egg-info" -or
        $Name -eq ".coverage"
    )
}

if (Test-Path -LiteralPath ".git") {
    $TrackedFiles = @(git ls-files)
    $TrackedHit = $TrackedFiles | Where-Object { Test-GeneratedArtifactPath $_ } | Select-Object -First 1
    if ($TrackedHit) {
        throw "Forbidden generated artifact is tracked: $TrackedHit"
    }
}

if ($CheckWorkingTreeArtifacts -or -not (Test-Path -LiteralPath ".git")) {
    foreach ($Pattern in $ForbiddenPatterns) {
        $Hit = Get-ChildItem -Path . -Recurse -Force -ErrorAction SilentlyContinue -Filter $Pattern |
            Where-Object {
                $Relative = Resolve-Path -LiteralPath $_.FullName -Relative
                $Relative = $Relative -replace "^\.[\\/]", ""
                Test-GeneratedArtifactPath $Relative
            } |
            Select-Object -First 1 -ExpandProperty FullName
        if ($Hit) {
            throw "Forbidden generated artifact is included: $Hit"
        }
    }
} elseif (Test-Path -LiteralPath ".git") {
    Write-Host "    working tree generated artifacts skipped (use -CheckWorkingTreeArtifacts to include untracked files)"
}

Write-Host "==> compile package smoke" -ForegroundColor Cyan
$Code = @'
from pathlib import Path
import sys

failed = False
for root in ("src", "tools"):
    for path in sorted(Path(root).rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except Exception as exc:
            failed = True
            print(f"{path}: {exc.__class__.__name__}: {exc}", file=sys.stderr)

raise SystemExit(1 if failed else 0)
'@
& $Python -c $Code
if ($LASTEXITCODE -ne 0) {
    throw "package smoke compile failed with exit code $LASTEXITCODE"
}
