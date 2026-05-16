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
