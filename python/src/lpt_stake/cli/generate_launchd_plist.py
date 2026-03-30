"""Generate a macOS launchd plist for the daily dashboard refresh."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape


def _default_paths() -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[4]
    python_dir = repo_root / "python"
    script_path = python_dir / "script" / "run-daily-refresh.sh"
    output_path = python_dir / "ops" / "org.livepeer.emissions-risk.refresh.plist"
    return python_dir, script_path, output_path


def parse_args() -> argparse.Namespace:
    python_dir, script_path, output_path = _default_paths()
    parser = argparse.ArgumentParser(
        description="Generate a launchd plist for the internal dashboard daily refresh."
    )
    parser.add_argument(
        "--label",
        default="org.livepeer.emissions-risk.refresh",
        help="launchd label to use in the plist.",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=6,
        help="Local hour for the daily refresh. Default: 6.",
    )
    parser.add_argument(
        "--minute",
        type=int,
        default=15,
        help="Local minute for the daily refresh. Default: 15.",
    )
    parser.add_argument(
        "--python-dir",
        type=Path,
        default=python_dir,
        help=f"Working directory for the refresh job. Default: {python_dir}",
    )
    parser.add_argument(
        "--script-path",
        type=Path,
        default=script_path,
        help=f"Runner script path. Default: {script_path}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=output_path,
        help=f"Where to write the plist. Default: {output_path}",
    )
    args = parser.parse_args()

    if not 0 <= args.hour <= 23:
        parser.error("--hour must be between 0 and 23.")
    if not 0 <= args.minute <= 59:
        parser.error("--minute must be between 0 and 59.")

    return args


def build_plist(label: str, hour: int, minute: int, python_dir: Path, script_path: Path) -> str:
    escaped_label = escape(label)
    escaped_python_dir = escape(str(python_dir))
    escaped_script_path = escape(str(script_path))

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{escaped_label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{escaped_script_path}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>{escaped_python_dir}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>{hour}</integer>
    <key>Minute</key>
    <integer>{minute}</integer>
  </dict>
</dict>
</plist>
"""


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        build_plist(
            label=args.label,
            hour=args.hour,
            minute=args.minute,
            python_dir=args.python_dir,
            script_path=args.script_path,
        )
    )

    print(f"Wrote launchd plist to {args.output_path}")
    print("To install it on macOS:")
    print(f"  cp {args.output_path} ~/Library/LaunchAgents/{args.label}.plist")
    print(f"  launchctl load ~/Library/LaunchAgents/{args.label}.plist")


if __name__ == "__main__":
    main()
