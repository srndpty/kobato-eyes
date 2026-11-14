# Repository-local development entry point. Leave arguments unbound so native
# tool flags and forwarded PowerShell switches remain intact.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Command = if ($args.Count) { ([string]$args[0]).ToLowerInvariant() } else { "help" }
$CommandArgs = @($args | Select-Object -Skip 1)
if ($CommandArgs.Count -and $CommandArgs[0] -eq "--") {
    $CommandArgs = @($CommandArgs | Select-Object -Skip 1)
}
$ProjectRoot = $PSScriptRoot

function Show-DevHelp {
    # Advertise only implemented commands.
    Write-Output @"
kobato-eyes development commands

Usage: .\dev.ps1 <command> [arguments...]
       dev <command> [arguments...]     (global dispatcher, from repo root)

  help                     Show this help (also: -h, --help).
  run / gui [arguments]    Start PyQt6; disable KOE_HEADLESS.
  build [arguments]        Build dist/kobato-eyes with the existing PyInstaller spec.
  test [unit] [pytest...]  Fast headless tests; no GUI/integration/GPU/db_stress.
  test gui [pytest...]    GUI / smoke tests with the real Qt backend.
  test integration [...]  Integration tests, excluding GPU.
  test gpu [pytest...]    Optional environment-dependent GPU tests.
  test db-stress [...]    SQLite concurrency / stress tests.
  test package [switches] Run scripts/check-package-smoke.ps1.
  lint [ruff arguments]   Ruff lint, then repository-wide format verification.
  check [switches]        Run scripts/check.ps1, including its coverage checks.

Examples:
  dev run
  dev test tests/core/test_query.py -q
  dev test gui -k settings
  dev lint --fix
  dev check -NoIncremental

Python: prefer .venv\Scripts\python.exe; otherwise python on PATH.
Extra arguments and exit codes are forwarded. Dependencies are not installed
automatically. clean is reserved until its deletion scope is defined.
"@
}

function Get-DevPython {
    # Activation is unnecessary: prefer the repository's virtualenv.
    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        return $VenvPython
    }
    $PythonCommand = Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $PythonCommand) {
        throw "Python not found. Create .venv first: python -m venv .venv"
    }
    return $PythonCommand.Source
}

function Invoke-DevScript {
    param([string]$Name, [object[]]$ForwardedArgs)
    # A child shell correctly binds forwarded switches and isolates environment
    # or working-directory changes made by the existing scripts.
    $Shell = if ($PSVersionTable.PSEdition -eq "Core") {
        Join-Path $PSHOME "pwsh.exe"
    } else {
        Join-Path $PSHOME "powershell.exe"
    }
    & $Shell -NoLogo -NoProfile -NonInteractive -File (Join-Path $ProjectRoot "scripts/$Name") @ForwardedArgs
    $script:ExitCode = $LASTEXITCODE
}

if ($Command -in @("help", "-h", "--help", "/?")) {
    Show-DevHelp
    exit 0
}
if ($Command -notin @("run", "gui", "build", "test", "lint", "check")) {
    [Console]::Error.WriteLine("Unsupported command '$Command'. Run dev help for available commands.")
    exit 2
}

$SavedEnvironment = @{}
foreach ($Name in @("PYTHONPATH", "KOE_HEADLESS")) {
    $SavedEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process")
}
$ExitCode = 1
Push-Location -LiteralPath $ProjectRoot
try {
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"
    switch ($Command) {
        { $_ -in @("run", "gui") } {
            Remove-Item Env:KOE_HEADLESS -ErrorAction SilentlyContinue
            & (Get-DevPython) -m ui.app @CommandArgs
            $ExitCode = $LASTEXITCODE
        }
        "build" {
            & (Get-DevPython) -m PyInstaller --clean --noconfirm `
                --workpath (Join-Path $ProjectRoot "tmp/pyinstaller-build") `
                --distpath (Join-Path $ProjectRoot "dist") `
                (Join-Path $ProjectRoot "tools/kobato-eyes.spec") @CommandArgs
            $ExitCode = $LASTEXITCODE
        }
        "check" {
            Invoke-DevScript "check.ps1" $CommandArgs
        }
        "lint" {
            & (Get-DevPython) -m ruff check @CommandArgs
            $ExitCode = $LASTEXITCODE
            if ($ExitCode -eq 0) {
                & (Get-DevPython) -m ruff format --check .
                $ExitCode = $LASTEXITCODE
            }
        }
        "test" {
            $Mode = "unit"
            $Markers = @{
                unit = "not (gui or smoke or integration or db_stress or gpu)"
                gui = "gui or smoke"
                integration = "integration and not gpu"
                gpu = "gpu"
                "db-stress" = "db_stress"
            }
            if ($CommandArgs.Count -and ([string]$CommandArgs[0]).ToLowerInvariant() -in @($Markers.Keys) + @("package")) {
                $Mode = ([string]$CommandArgs[0]).ToLowerInvariant()
                $CommandArgs = @($CommandArgs | Select-Object -Skip 1)
            }
            if ($Mode -eq "package") {
                Invoke-DevScript "check-package-smoke.ps1" $CommandArgs
            } else {
                if ($Mode -eq "unit" -or $Mode -eq "gpu") {
                    $env:KOE_HEADLESS = "1"
                } else {
                    Remove-Item Env:KOE_HEADLESS -ErrorAction SilentlyContinue
                }
                & (Get-DevPython) -m pytest -m $Markers[$Mode] -p no:cov @CommandArgs
                $ExitCode = $LASTEXITCODE
            }
        }
    }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    $ExitCode = 1
} finally {
    foreach ($Name in $SavedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $SavedEnvironment[$Name], "Process")
    }
    Pop-Location
}
exit $ExitCode
