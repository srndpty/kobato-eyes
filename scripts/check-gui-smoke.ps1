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

Write-Host "==> pytest gui or smoke" -ForegroundColor Cyan
& $Python -m pytest -m "gui or smoke" -p no:cov
if ($LASTEXITCODE -ne 0) {
    throw "pytest gui/smoke failed with exit code $LASTEXITCODE"
}
