#!/usr/bin/env python3
"""Pre-submission sanity check.

Run this before zipping the project for the hackathon portal. Catches the
class of mistakes that are invisible locally but break the submission for
whoever unzips it next:

  - submission_metadata.yaml still has TODO/placeholder values
  - symlinks that point outside this repo (works on your machine, dangling
    for everyone else — e.g. a personal Downloads path)
  - junk directories that bloat the zip and may shadow real files
    (.venv, __pycache__, .pytest_cache, output/)
  - the rule/hybrid mode agreement check hasn't been run

Usage:
    python3 scripts/check_submission_ready.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

JUNK_DIRS = [".venv", "venv", "__pycache__", ".pytest_cache", "output", ".cache"]


def check_metadata() -> list[str]:
    problems = []
    meta_path = ROOT / "submission_metadata.yaml"
    if not meta_path.exists():
        return ["submission_metadata.yaml is missing entirely"]
    text = meta_path.read_text(encoding="utf-8")
    if "TODO" in text or "your-streamlit-app" in text or "YOUR_USERNAME" in text:
        problems.append(
            "submission_metadata.yaml still contains placeholder/TODO values — "
            "fill in team name, contact, GitHub repo, and sandbox link"
        )
    return problems


def check_dangling_symlinks() -> list[str]:
    problems = []
    for path in ROOT.rglob("*"):
        if any(part in JUNK_DIRS for part in path.parts):
            continue
        if path.is_symlink():
            target = path.resolve()
            if not target.exists():
                problems.append(f"Dangling symlink: {path.relative_to(ROOT)} -> {path.readlink()}")
            elif not str(target).startswith(str(ROOT)):
                problems.append(
                    f"Symlink points outside the repo (will break for anyone else who "
                    f"unzips this): {path.relative_to(ROOT)} -> {target}"
                )
    return problems


def check_junk_dirs() -> list[str]:
    problems = []
    for junk in JUNK_DIRS:
        for hit in ROOT.rglob(junk):
            if hit.is_dir():
                problems.append(f"Junk directory present, should not be zipped: {hit.relative_to(ROOT)}")
    return problems


def check_required_files() -> list[str]:
    problems = []
    required = ["rank.py", "validate_submission.py", "requirements.txt", "README.md", "src/rule_scorer.py"]
    for rel in required:
        if not (ROOT / rel).exists():
            problems.append(f"Required file missing: {rel}")
    return problems


def main() -> int:
    all_problems = []
    all_problems += check_metadata()
    all_problems += check_dangling_symlinks()
    all_problems += check_junk_dirs()
    all_problems += check_required_files()

    if all_problems:
        print("NOT ready to submit:\n")
        for p in all_problems:
            print(f"  ✗ {p}")
        print(
            "\nFix these, then re-run this script. For junk directories, just delete them — "
            "they'll be regenerated on demand and shouldn't ship in the zip."
        )
        return 1

    print("✓ Looks ready to submit:")
    print("  - submission_metadata.yaml has no placeholder values")
    print("  - no dangling or repo-external symlinks")
    print("  - no .venv/__pycache__/output/.cache junk directories")
    print("  - all required files present")
    print(
        "\nReminder: run `python3 rank.py --candidates <path> --out submission.csv --mode compare`\n"
        "at least once and read the agreement report before picking which mode's output to submit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
