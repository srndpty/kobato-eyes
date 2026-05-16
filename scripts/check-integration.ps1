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
Remove-Item Env:KOE_HEADLESS -ErrorAction SilentlyContinue

Write-Host "==> pytest integration and not gpu" -ForegroundColor Cyan
& $Python -m pytest -m "integration and not gpu" -p no:cov
if ($LASTEXITCODE -ne 0) {
    throw "pytest integration failed with exit code $LASTEXITCODE"
}
