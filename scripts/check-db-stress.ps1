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
$env:KOE_HEADLESS = "1"

Write-Host "==> pytest db_stress" -ForegroundColor Cyan
& $Python -m pytest -m "db_stress" -q
if ($LASTEXITCODE -ne 0) {
    throw "pytest db_stress failed with exit code $LASTEXITCODE"
}
