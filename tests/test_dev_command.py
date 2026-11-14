"""Windows development command routing without starting the app or packaging it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SHELL = shutil.which("pwsh") or shutil.which("powershell")
pytestmark = pytest.mark.skipif(sys.platform != "win32" or SHELL is None, reason="Windows PowerShell required")


@pytest.fixture
def dev_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Use harmless fake Python entry points to record actual native argument passing."""
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "dev.ps1", repo / "dev.ps1")
    source = repo / "src"
    source.mkdir()
    probe = """import json, os, sys
from pathlib import Path
with Path(os.environ["DEV_PROBE"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(dict(args=sys.argv, cwd=os.getcwd(), pythonpath=os.getenv("PYTHONPATH"),
                                headless=os.getenv("KOE_HEADLESS"))) + "\\n")
raise SystemExit(int(os.getenv("DEV_PROBE_EXIT", "0")))
"""
    for module in ("pytest.py", "ruff.py", "PyInstaller.py"):
        (source / module).write_text(probe, encoding="utf-8")
    (source / "ui").mkdir()
    (source / "ui/__init__.py").touch()
    (source / "ui/app.py").write_text(probe, encoding="utf-8")
    env = dict(os.environ, DEV_PROBE=str(repo / "probe.jsonl"), KOE_HEADLESS="caller", PYTHONPATH="caller")
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return repo, env


def invoke(repo: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    """Call the real script, preserving native argument boundaries."""
    return subprocess.run(
        [SHELL, "-NoProfile", "-File", str(repo / "dev.ps1"), *args],
        cwd=repo,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def read_probe(repo: Path) -> list[dict]:
    """Read arguments and environment observed by the fake command."""
    return [json.loads(line) for line in (repo / "probe.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    "mode,headless,marker",
    [
        ("unit", "1", "not (gui or smoke or integration or db_stress or gpu)"),
        ("gui", None, "gui or smoke"),
        ("integration", None, "integration and not gpu"),
    ],
)
def test_test_modes_and_quoted_arguments(dev_repo, mode, headless, marker) -> None:
    """Mode selection controls Qt backend; a spaced pytest expression remains one argument."""
    repo, env = dev_repo
    result = invoke(repo, env, "test", mode, "-k", "one or two", "-q")
    assert result.returncode == 0, result.stderr
    record = read_probe(repo)[0]
    assert record["args"][1:] == ["-m", marker, "-p", "no:cov", "-k", "one or two", "-q"]
    assert record["headless"] == headless
    assert Path(record["pythonpath"]) == repo / "src"


def test_run_restores_caller_and_propagates_exit(dev_repo) -> None:
    """GUI entry disables headless in the child and restores the calling PowerShell session."""
    repo, env = dev_repo
    env["DEV_PROBE_EXIT"] = "7"
    (repo / "caller").mkdir()
    command = """$before = (Get-Location).Path
& ../dev.ps1 run "argument with spaces"
$code = $LASTEXITCODE
if ($env:PYTHONPATH -ne "caller" -or $env:KOE_HEADLESS -ne "caller" -or (Get-Location).Path -ne $before) { exit 99 }
exit $code
"""
    result = subprocess.run(
        [SHELL, "-NoProfile", "-Command", command],
        cwd=repo / "caller",
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 7, result.stderr
    record = read_probe(repo)[0]
    assert record["headless"] is None
    assert Path(record["cwd"]) == repo
    assert record["args"][1:] == ["argument with spaces"]


def test_check_forwards_powershell_switch_and_output(dev_repo) -> None:
    """Delegated PS scripts receive switches, retain their output and propagate failure."""
    repo, env = dev_repo
    (repo / "scripts").mkdir()
    (repo / "scripts/check.ps1").write_text(
        'param([switch]$NoCoverage, [string]$CoverageFile)\nWrite-Output "$NoCoverage|$CoverageFile"\nexit 7\n',
        encoding="utf-8",
    )
    result = invoke(repo, env, "check", "-NoCoverage", "-CoverageFile", "path with spaces")
    assert result.returncode == 7, result.stderr
    assert "True|path with spaces" in result.stdout


def test_help_and_unsupported_commands_do_not_launch_tools(dev_repo) -> None:
    """The command list does not silently implement destructive clean behavior."""
    repo, env = dev_repo
    assert invoke(repo, env, "help").returncode == 0
    result = invoke(repo, env, "clean")
    assert result.returncode == 2 and "Unsupported command" in result.stderr
    assert not (repo / "probe.jsonl").exists()


@pytest.mark.parametrize("command", ["gui", "build", "lint"])
def test_remaining_python_commands_are_routed(dev_repo, command) -> None:
    """Build uses the spec and lint also checks formatting, without running either real tool."""
    repo, env = dev_repo
    result = invoke(repo, env, command)
    assert result.returncode == 0, result.stderr
    records = read_probe(repo)
    if command == "gui":
        assert records[0]["headless"] is None
    elif command == "build":
        assert "--distpath" in records[0]["args"]
        assert str(repo / "tools/kobato-eyes.spec") in records[0]["args"]
    else:
        assert [record["args"][1:] for record in records] == [["check"], ["format", "--check", "."]]
