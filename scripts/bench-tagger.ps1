param(
    [Parameter(Mandatory = $true)]
    [string] $Root,

    [Parameter(Mandatory = $true)]
    [string] $Model,

    [string] $TagsCsv = "",
    [string] $Tagger = "wd14-onnx",
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
    [string[]] $Extension = @(),
    [switch] $InputCache,
    [string] $InputCacheDir = "",
    [Alias("InputCacheExtensions")]
    [string[]] $InputCacheExtension = @(),
    [string] $Output = "tmp\bench\tagger-bench.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $projectRoot "src"

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$scriptPath = Join-Path $projectRoot "scripts\bench_tagger.py"
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    $Output
} else {
    Join-Path $projectRoot $Output
}

$argsList = @(
    $scriptPath,
    "--root", $Root,
    "--model", $Model,
    "--tagger", $Tagger,
    "--provider", $Provider,
    "--device", $Device,
    "--batch-size", "$BatchSize",
    "--limit", "$Limit",
    "--warmup-batches", "$WarmupBatches",
    "--prefetch-depth", "$PrefetchDepth",
    "--output", $outputPath
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
foreach ($ext in $Extension) {
    if ($ext) {
        $argsList += @("--extension", $ext)
    }
}
if ($InputCache) {
    $argsList += "--input-cache"
}
if ($InputCacheDir) {
    $argsList += @("--input-cache-dir", $InputCacheDir)
}
foreach ($ext in $InputCacheExtension) {
    if ($ext) {
        $argsList += @("--input-cache-extension", $ext)
    }
}

& $python @argsList
