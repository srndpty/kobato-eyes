param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^v?[0-9]+(\.[0-9]+){1,2}([-.][A-Za-z0-9.]+)?$')]
    [string]$Version,

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$Platform = "win-x64",

    [ValidateRange(0, 9)]
    [int]$CompressionLevel = 9,

    [switch]$Clean,
    [switch]$SkipBuild,
    [switch]$Zip
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$VersionLabel = if ($Version.StartsWith("v", [StringComparison]::OrdinalIgnoreCase)) {
    $Version
} else {
    "v$Version"
}

$DistRoot = Join-Path $ProjectRoot "dist"
$AppDir = Join-Path $DistRoot "kobato-eyes"
$ReleaseDir = Join-Path $DistRoot "release"
$BuildWorkDir = Join-Path $ProjectRoot "tmp\pyinstaller-build"
$SpecPath = Join-Path $ProjectRoot "tools\kobato-eyes.spec"
$ArchiveBaseName = "kobato-eyes-$VersionLabel-$Platform"
$SevenZipPath = Join-Path $ReleaseDir "$ArchiveBaseName.7z"
$ZipPath = Join-Path $ReleaseDir "$ArchiveBaseName.zip"
$ChecksumsPath = Join-Path $ReleaseDir "SHA256SUMS.txt"

function Assert-InProject {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $FullPath = [System.IO.Path]::GetFullPath($Path)
    $RootPath = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    $RootWithSeparator = $RootPath + [System.IO.Path]::DirectorySeparatorChar
    if (
        -not $FullPath.Equals($RootPath, [StringComparison]::OrdinalIgnoreCase) -and
        -not $FullPath.StartsWith($RootWithSeparator, [StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "Refusing to operate outside project root: $FullPath"
    }
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Resolve-Executable {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Candidates
    )

    foreach ($Candidate in $Candidates) {
        if ([System.IO.Path]::IsPathRooted($Candidate)) {
            if (Test-Path -LiteralPath $Candidate) {
                return $Candidate
            }
            continue
        }

        $Command = Get-Command $Candidate -ErrorAction SilentlyContinue
        if ($Command) {
            return $Command.Source
        }
    }

    return $null
}

function Write-ChecksumFile {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Paths
    )

    $Lines = foreach ($Path in $Paths) {
        if (-not (Test-Path -LiteralPath $Path)) {
            continue
        }

        $Hash = Get-FileHash -LiteralPath $Path -Algorithm SHA256
        "$($Hash.Hash.ToLowerInvariant())  $(Split-Path -Leaf $Path)"
    }

    Set-Content -LiteralPath $ChecksumsPath -Value $Lines -Encoding ascii
}

Assert-InProject $DistRoot
Assert-InProject $ReleaseDir
Assert-InProject $BuildWorkDir
Assert-InProject $SevenZipPath
Assert-InProject $ZipPath
Assert-InProject $ChecksumsPath

if (-not (Test-Path -LiteralPath $SpecPath)) {
    throw "PyInstaller spec was not found: $SpecPath"
}

if ($Clean) {
    foreach ($Path in @($BuildWorkDir, $AppDir, $SevenZipPath, $ZipPath, $ChecksumsPath)) {
        Assert-InProject $Path
        if (Test-Path -LiteralPath $Path) {
            Remove-Item -LiteralPath $Path -Recurse -Force
        }
    }
}

if (-not (Test-Path -LiteralPath $ReleaseDir)) {
    New-Item -ItemType Directory -Path $ReleaseDir | Out-Null
}

if (-not $SkipBuild) {
    $VenvPyInstaller = Join-Path $ProjectRoot ".venv\Scripts\pyinstaller.exe"
    $PyInstaller = Resolve-Executable @($VenvPyInstaller, "pyinstaller")
    if (-not $PyInstaller) {
        throw "PyInstaller was not found. Install packaging dependencies first."
    }

    Invoke-Step "PyInstaller build" {
        & $PyInstaller `
            --clean `
            --noconfirm `
            --workpath $BuildWorkDir `
            --distpath $DistRoot `
            $SpecPath
    }
}

if (-not (Test-Path -LiteralPath $AppDir)) {
    throw "Application directory was not found: $AppDir"
}

$SevenZipCandidates = @("7z", "7zz")
if ($env:ProgramFiles) {
    $SevenZipCandidates += Join-Path $env:ProgramFiles "7-Zip\7z.exe"
}
if (${env:ProgramFiles(x86)}) {
    $SevenZipCandidates += Join-Path ${env:ProgramFiles(x86)} "7-Zip\7z.exe"
}

$SevenZip = Resolve-Executable $SevenZipCandidates
if (-not $SevenZip) {
    throw "7-Zip was not found. Install 7-Zip or add 7z.exe to PATH."
}

if (Test-Path -LiteralPath $SevenZipPath) {
    Remove-Item -LiteralPath $SevenZipPath -Force
}

Invoke-Step "7z archive ($CompressionLevel)" {
    & $SevenZip `
        a `
        -t7z `
        $SevenZipPath `
        (Join-Path $AppDir "*") `
        "-mx=$CompressionLevel" `
        -mmt=on
}

$PackagePaths = @($SevenZipPath)

if ($Zip) {
    if (Test-Path -LiteralPath $ZipPath) {
        Remove-Item -LiteralPath $ZipPath -Force
    }

    Invoke-Step "zip archive" {
        Compress-Archive `
            -Path (Join-Path $AppDir "*") `
            -DestinationPath $ZipPath `
            -CompressionLevel Optimal
    }
    $PackagePaths += $ZipPath
}

Write-ChecksumFile -Paths $PackagePaths

Write-Host ""
Write-Host "Release package complete:" -ForegroundColor Green
foreach ($Path in $PackagePaths) {
    $Item = Get-Item -LiteralPath $Path
    $SizeGiB = $Item.Length / 1GB
    Write-Host ("  {0} ({1:N2} GiB)" -f $Path, $SizeGiB)
}
Write-Host "  $ChecksumsPath"
