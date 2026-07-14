from __future__ import annotations

import re
import sys
from importlib.metadata import distributions
from pathlib import Path

from pip._vendor.distlib.scripts import ScriptMaker


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = (ROOT / ".venv").resolve()
SCRIPTS_DIR = VENV_DIR / "Scripts"


def _write_project_path_file() -> None:
    site_packages = VENV_DIR / "Lib" / "site-packages"
    site_packages.mkdir(parents=True, exist_ok=True)
    (site_packages / "ocr_demo_project_root.pth").write_text(
        f"{ROOT}\n",
        encoding="utf-8",
        newline="\n",
    )


def _rewrite_activation_scripts() -> None:
    replacements = {
        SCRIPTS_DIR / "activate.bat": (
            re.compile(r"(?m)^set VIRTUAL_ENV=.*$"),
            f"set VIRTUAL_ENV={VENV_DIR}",
        ),
        SCRIPTS_DIR / "activate": (
            re.compile(r'export VIRTUAL_ENV=\$\(cygpath "[^"]*"\)'),
            f'export VIRTUAL_ENV=$(cygpath "{VENV_DIR}")',
        ),
    }
    for path, (pattern, replacement) in replacements.items():
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        content = pattern.sub(lambda _match: replacement, content)
        if path.name == "activate":
            content = re.sub(
                r'export VIRTUAL_ENV="[^"]*"',
                lambda _match: f'export VIRTUAL_ENV="{VENV_DIR}"',
                content,
            )
        path.write_text(content, encoding="utf-8", newline="\n")


def _rewrite_text_entrypoint_shebangs() -> int:
    """Repair extensionless/script-style launchers left behind after a venv move."""
    rewritten = 0
    current = str(Path(sys.executable).resolve()).encode("utf-8")
    for path in SCRIPTS_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() in {".exe", ".dll", ".pyd"}:
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        first_line, separator, remainder = content.partition(b"\n")
        if not first_line.startswith(b"#!") or b"python" not in first_line.lower():
            continue
        expected = b"#!" + current
        if first_line.rstrip(b"\r") == expected:
            continue
        newline = b"\r\n" if first_line.endswith(b"\r") else b"\n"
        path.write_bytes(expected + (newline + remainder if separator else b""))
        rewritten += 1
    return rewritten


def _rebuild_entrypoints() -> tuple[int, list[str]]:
    entries: dict[tuple[str, str], str] = {}
    for distribution in distributions():
        for entry in distribution.entry_points:
            if entry.group in {"console_scripts", "gui_scripts"}:
                entries[(entry.group, entry.name)] = entry.value

    # Avoid importing ModelScope/Torch before Paddle on Windows; that ordering
    # can make PaddleOCR's CUDA DLLs fail to initialize.
    if ("console_scripts", "paddleocr") in entries:
        entries[("console_scripts", "paddleocr")] = "app.ocr_engines.paddleocr_cli:main"

    maker = ScriptMaker(None, str(SCRIPTS_DIR))
    maker.executable = sys.executable
    maker.clobber = True
    maker.variants = {""}
    errors: list[str] = []
    written = 0
    for (group, name), value in sorted(entries.items()):
        try:
            written += len(
                maker.make(
                    f"{name} = {value}",
                    options={"gui": group == "gui_scripts"},
                )
            )
        except Exception as exc:  # pragma: no cover - diagnostic output
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    pip_alias = f"pip{sys.version_info.major}.{sys.version_info.minor}"
    try:
        written += len(maker.make(f"{pip_alias} = pip._internal.cli.main:main"))
    except Exception as exc:  # pragma: no cover - diagnostic output
        errors.append(f"{pip_alias}: {type(exc).__name__}: {exc}")
    return written, errors


def main() -> int:
    if Path(sys.prefix).resolve() != VENV_DIR:
        print(f"Run this script with {SCRIPTS_DIR / 'python.exe'}", file=sys.stderr)
        return 2
    _write_project_path_file()
    _rewrite_activation_scripts()
    written, errors = _rebuild_entrypoints()
    rewritten = _rewrite_text_entrypoint_shebangs()
    print(f"Rebuilt {written} launchers for {sys.executable}")
    print(f"Repaired {rewritten} text launcher shebangs")
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
