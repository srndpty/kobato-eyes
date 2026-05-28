param()

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

if (Test-Path -LiteralPath ".git") {
    $TrackedFiles = @(git ls-files)
    $TrackedHit = $TrackedFiles | Where-Object {
        $Name = Split-Path -Leaf $_
        $Normalized = $_ -replace "\\", "/"
        $Normalized -like "__pycache__/*" -or
        $Normalized -like "*/__pycache__/*" -or
        $Normalized -like ".pytest_cache/*" -or
        $Normalized -like "*/.pytest_cache/*" -or
        $Normalized -like "*.egg-info/*" -or
        $Normalized -like "*/*.egg-info/*" -or
        $Name -like "*.pyc" -or
        $Name -like "*.pyo" -or
        $Name -eq ".coverage"
    } | Select-Object -First 1
    if ($TrackedHit) {
        throw "Forbidden generated artifact is tracked: $TrackedHit"
    }
} else {
    foreach ($Pattern in $ForbiddenPatterns) {
        $Hit = Get-ChildItem -Path . -Recurse -Force -ErrorAction SilentlyContinue -Filter $Pattern |
            Select-Object -First 1 -ExpandProperty FullName
        if ($Hit) {
            throw "Forbidden generated artifact is included: $Hit"
        }
    }
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
