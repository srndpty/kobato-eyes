param(
    [Parameter(Mandatory = $true)]
    [string] $Root,

    [Parameter(Mandatory = $true)]
    [string] $Model,

    [string] $TagsCsv = "",
    [ValidateSet("auto", "wd14", "pixai")]
    [string] $Provider = "auto",
    [ValidateSet("auto", "tensorrt", "cuda", "cpu")]
    [string] $Device = "auto",
    [int] $BatchSize = 32,
    [int] $Limit = 1000,
    [int] $WarmupBatches = 2,
    [int] $PrefetchDepth = 4,
    [int] $IoWorkers = 0,
    [int] $TopkCap = 0,
    [string] $Output = "tmp\bench\tagger-bench.json"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$argsList = @(
    "scripts\bench_tagger.py",
    "--root", $Root,
    "--model", $Model,
    "--provider", $Provider,
    "--device", $Device,
    "--batch-size", "$BatchSize",
    "--limit", "$Limit",
    "--warmup-batches", "$WarmupBatches",
    "--prefetch-depth", "$PrefetchDepth",
    "--output", $Output
)

if ($TagsCsv) {
    $argsList += @("--tags-csv", $TagsCsv)
}
if ($IoWorkers -gt 0) {
    $argsList += @("--io-workers", "$IoWorkers")
}
if ($TopkCap -gt 0) {
    $argsList += @("--topk-cap", "$TopkCap")
}

& $python @argsList
