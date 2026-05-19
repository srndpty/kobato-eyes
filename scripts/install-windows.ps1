param(
    [switch]$Build,
    [string]$Destination = (Join-Path $env:ProgramFiles "kobato-eyes")
)

$ErrorActionPreference = "Stop"

$Script:AppName = "kobato-eyes"
$Script:ExeName = "kobato-eyes.exe"

$Script:DirectorySeparators = @(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
)

# 権限確認と引数クォート用の小さなヘルパー。
function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal -ArgumentList $identity
    return ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))
}

function ConvertTo-Argument([string]$Value) {
    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return ('"' + $escaped + '"')
}

function ConvertTo-FullDirectoryPath([string]$Path) {
    return ([System.IO.Path]::GetFullPath($Path).TrimEnd($Script:DirectorySeparators))
}

function Resolve-Executable {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Candidates
    )

    foreach ($candidate in $Candidates) {
        if ([System.IO.Path]::IsPathRooted($candidate)) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return $candidate
            }
            continue
        }

        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    return $null
}

# インストール先の安全確認。宛先の中身を削除するため厳しめに判定する。
function Test-DirectChildPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Parent
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd($Script:DirectorySeparators)
    $fullParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd($Script:DirectorySeparators)
    $actualParent = Split-Path -Parent $fullPath
    return ([string]::Equals($actualParent, $fullParent, [StringComparison]::OrdinalIgnoreCase))
}

function Test-KobatoEyesInstallDirectoryName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $leafName = Split-Path -Leaf ([System.IO.Path]::GetFullPath($Path).TrimEnd($Script:DirectorySeparators))
    return (
        $leafName -eq $Script:AppName -or
        $leafName.StartsWith("$($Script:AppName)-", [StringComparison]::OrdinalIgnoreCase)
    )
}

function Test-OnedirInstall {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return (
        (Test-Path -LiteralPath (Join-Path $Path $Script:ExeName) -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path "_internal") -PathType Container)
    )
}

function Test-DirectoryEmpty {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $firstChild = Get-ChildItem -LiteralPath $Path -Force -ErrorAction Stop | Select-Object -First 1
    return ($null -eq $firstChild)
}

function Test-ReparsePoint {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    return ($item -and (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0))
}

function Test-KobatoEyesProcessInDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $installRoot = (
        [System.IO.Path]::GetFullPath($Path).TrimEnd($Script:DirectorySeparators) +
        [System.IO.Path]::DirectorySeparatorChar
    )

    foreach ($process in Get-Process -Name $Script:AppName -ErrorAction SilentlyContinue) {
        try {
            $modulePath = [System.IO.Path]::GetFullPath($process.MainModule.FileName)
        } catch {
            return $true
        }
        if ($modulePath.StartsWith($installRoot, [StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Copy-DirectoryContents {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,

        [Parameter(Mandatory = $true)]
        [string]$Destination,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-Verbose "Copying $Description from $Source to $Destination"
    try {
        foreach ($item in Get-ChildItem -LiteralPath $Source -Force) {
            Copy-Item -LiteralPath $item.FullName -Destination $Destination -Recurse -Force
        }
    } catch {
        throw ("Failed to copy {0} from {1} to {2}: {3}" -f $Description, $Source, $Destination, $_.Exception.Message)
    }
}

function Clear-DirectoryContents {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (Test-Path -LiteralPath $Path) {
        Get-ChildItem -LiteralPath $Path -Force |
            Remove-Item -Recurse -Force
    } else {
        [System.IO.Directory]::CreateDirectory($Path) | Out-Null
    }
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$DistRoot = Join-Path $RepoRoot "dist"
$Source = Join-Path $DistRoot $Script:AppName
$BuildWorkDir = Join-Path $RepoRoot "tmp\pyinstaller-build"
$SpecPath = Join-Path $RepoRoot "tools\kobato-eyes.spec"
$DestinationFullPath = ConvertTo-FullDirectoryPath $Destination
$ExpectedProgramFilesRoot = ConvertTo-FullDirectoryPath $env:ProgramFiles

# 昇格を求める前に、広すぎる宛先や無関係な宛先を拒否する。
if (-not (Test-DirectChildPath -Path $DestinationFullPath -Parent $ExpectedProgramFilesRoot)) {
    throw "Destination must be a direct child directory under Program Files: $DestinationFullPath"
}
if (-not (Test-KobatoEyesInstallDirectoryName -Path $DestinationFullPath)) {
    throw "Destination directory name must be kobato-eyes or kobato-eyes-*: $DestinationFullPath"
}

# ビルドは通常ユーザーの venv/PATH で実行し、コピーだけを昇格する。
if ($Build) {
    Push-Location $RepoRoot
    try {
        if (-not (Test-Path -LiteralPath $SpecPath -PathType Leaf)) {
            throw "PyInstaller spec was not found: $SpecPath"
        }

        $venvPyInstaller = Join-Path $RepoRoot ".venv\Scripts\pyinstaller.exe"
        $pyInstaller = Resolve-Executable @($venvPyInstaller, "pyinstaller")
        if (-not $pyInstaller) {
            throw "PyInstaller was not found. Install packaging dependencies first: python -m pip install -c requirements-dev.lock -e `".[dev,tagging-gpu,packaging]`""
        }

        Write-Host "Building kobato-eyes..."
        & $pyInstaller `
            --clean `
            --noconfirm `
            --workpath $BuildWorkDir `
            --distpath $DistRoot `
            $SpecPath
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

# Program Files へ書き込むため、未昇格なら管理者として再実行する。
if (-not (Test-Administrator)) {
    $argsList = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        (ConvertTo-Argument $PSCommandPath),
        "-Destination",
        (ConvertTo-Argument $DestinationFullPath)
    )
    if ($VerbosePreference -ne "SilentlyContinue") {
        $argsList += "-Verbose"
    }
    $process = Start-Process -FilePath "powershell" -ArgumentList $argsList -Verb RunAs -Wait -PassThru
    exit $process.ExitCode
}

# 検証済みのステージングからインストール先を置き換える。
Push-Location $RepoRoot
try {
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Build output was not found: $Source. Run .\scripts\install-windows.ps1 -Build, or build with .\scripts\package-release.ps1 first."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Source $Script:ExeName) -PathType Leaf)) {
        throw "Build output was not found: $Source. Run .\scripts\install-windows.ps1 -Build, or build with .\scripts\package-release.ps1 first."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Source "_internal") -PathType Container)) {
        throw "Build output is incomplete: $Source\_internal was not found."
    }

    Write-Host "Installing from $Source"
    Write-Host "Installing to   $DestinationFullPath"

    $destinationParent = Split-Path -Parent $DestinationFullPath
    $destinationLeaf = Split-Path -Leaf $DestinationFullPath
    $replaceId = [System.Guid]::NewGuid().ToString("N")
    $stagingDestination = Join-Path $destinationParent "$destinationLeaf.installing.$replaceId"
    $backupDestination = Join-Path $destinationParent "$destinationLeaf.backup.$replaceId"

    if (Test-Path -LiteralPath $DestinationFullPath -PathType Leaf) {
        throw "Destination exists but is not a directory: $DestinationFullPath"
    }
    if (Test-ReparsePoint -Path $DestinationFullPath) {
        throw "Destination must not be a reparse point: $DestinationFullPath"
    }
    if (Test-KobatoEyesProcessInDirectory -Path $DestinationFullPath) {
        throw "kobato-eyes is running from $DestinationFullPath. Close it before installing."
    }

    $backupCreated = $false
    $backupReady = $false
    $keepBackup = $false

    try {
        # 今回の実行で使う一時ディレクトリだけを事前に掃除する。
        Remove-Item -LiteralPath $stagingDestination -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $backupDestination -Recurse -Force -ErrorAction SilentlyContinue

        [System.IO.Directory]::CreateDirectory($stagingDestination) | Out-Null
        Copy-DirectoryContents -Source $Source -Destination $stagingDestination -Description "build output to staging"

        # onedir 配置として使えないコピー結果なら、既存インストールを触る前に失敗させる。
        if (-not (Test-OnedirInstall -Path $stagingDestination)) {
            throw "Install failed: staging directory is incomplete: $stagingDestination"
        }

        $hasExistingInstall = Test-Path -LiteralPath $DestinationFullPath -PathType Container
        if ($hasExistingInstall) {
            if ((-not (Test-OnedirInstall -Path $DestinationFullPath)) -and
                (-not (Test-DirectoryEmpty -Path $DestinationFullPath))) {
                throw "Destination exists but does not look like a kobato-eyes onedir install: $DestinationFullPath"
            }

            [System.IO.Directory]::CreateDirectory($backupDestination) | Out-Null
            $backupCreated = $true
            Copy-DirectoryContents -Source $DestinationFullPath -Destination $backupDestination -Description "existing install to backup"
            $backupReady = $true
        }

        try {
            Clear-DirectoryContents -Path $DestinationFullPath
            Copy-DirectoryContents -Source $stagingDestination -Destination $DestinationFullPath -Description "staging to install destination"

            if (-not (Test-OnedirInstall -Path $DestinationFullPath)) {
                throw "Install failed: destination directory is incomplete: $DestinationFullPath"
            }
        } catch {
            if ($backupReady) {
                Write-Warning "Install failed. Restoring previous install from $backupDestination"
                try {
                    Clear-DirectoryContents -Path $DestinationFullPath
                    Copy-DirectoryContents -Source $backupDestination -Destination $DestinationFullPath -Description "backup to install destination"
                } catch {
                    $keepBackup = $true
                    Write-Warning "Rollback failed. Backup remains at $backupDestination"
                    Write-Warning "Restore manually by copying the backup directory contents to $DestinationFullPath"
                }
            }
            throw
        }

        Write-Host "Installed kobato-eyes to $DestinationFullPath"
    } finally {
        Remove-Item -LiteralPath $stagingDestination -Recurse -Force -ErrorAction SilentlyContinue
        if ($backupCreated -and -not $keepBackup) {
            Remove-Item -LiteralPath $backupDestination -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

} finally {
    Pop-Location
}
