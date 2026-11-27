"""Deterministic repository hygiene checks, run in CI.

    python scripts/check_repo.py

Checks every git-tracked text file for unresolved merge-conflict markers, and
every tracked Markdown file for prose lines that were pasted twice and for
"committed" / "ignored" claims about a path that git disagrees with.
Exits non-zero and prints ``path:line: message`` for each problem.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFLICT_EDGE = re.compile(r"^(<{7}|>{7})(?: |$)")
CONFLICT_MIDDLE = re.compile(r"^={7}$")  # also a valid Markdown heading underline
MIN_LINE_CHARS = 40  # short repeats ("- `video_id`", "```sh") are normal


def tracked_files(root: Path = ROOT) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [root / name for name in out.decode().split("\0") if name]


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def conflict_markers(text: str) -> list[tuple[int, str]]:
    lines = text.splitlines()
    edges = [n for n, line in enumerate(lines, 1) if CONFLICT_EDGE.match(line)]
    middles = [n for n, line in enumerate(lines, 1) if CONFLICT_MIDDLE.match(line)] if edges else []
    return [(n, "unresolved merge-conflict marker") for n in sorted(edges + middles)]


def duplicated_markdown(text: str) -> list[tuple[int, str]]:
    """Prose lines that appear twice in a file, and identical adjacent lines.

    Pasted-twice sections rarely line up on paragraph boundaries, so this
    compares individual lines. Repeats inside fenced code blocks (shared
    command-line arguments, for example) are allowed unless adjacent.
    """
    problems = []
    seen: dict[str, int] = {}
    in_fence = False
    previous = ""
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            previous = ""
            continue
        if len(line) >= MIN_LINE_CHARS and not set(line) <= set("|-: "):
            if line == previous:
                problems.append((n, "line repeats the previous line"))
            elif not in_fence:
                if line in seen:
                    problems.append((n, f"line duplicates line {seen[line]}"))
                else:
                    seen[line] = n
        previous = line
    return problems


DOC_PATH = re.compile(r"`([^`\s<>*]+/[^`\s<>*]*)`")


def path_claims(text: str) -> list[tuple[int, str, str]]:
    """(line, path, claim) for lines that name one repo path and call it committed or ignored."""
    claims = []
    for n, line in enumerate(text.splitlines(), 1):
        paths = [p for p in DOC_PATH.findall(line) if not p.startswith(("-", "http", "~", "/"))]
        if len(paths) != 1:
            continue
        low = line.lower()
        if "not committed" in low:
            claim = "untracked"
        elif re.search(r"\bcommitted\b|\bchecked[- ]in\b", low):
            claim = "tracked"
        elif re.search(r"\bignored\b", low):
            claim = "ignored"
        else:
            continue
        claims.append((n, paths[0], claim))
    return claims


def git_tracked(path: str, tracked: set[str]) -> bool:
    path = path.rstrip("/")
    return path in tracked or any(name.startswith(path + "/") for name in tracked)


def git_ignored(path: str, root: Path = ROOT) -> bool:
    probe = path.rstrip("/") + ("/_" if path.endswith("/") else "")
    return subprocess.run(["git", "check-ignore", "-q", "--no-index", probe], cwd=root).returncode == 0


def claim_problems(text: str, tracked: set[str], root: Path = ROOT) -> list[tuple[int, str]]:
    problems = []
    for n, path, claim in path_claims(text):
        is_tracked = git_tracked(path, tracked)
        if claim == "tracked" and not is_tracked:
            problems.append((n, f"`{path}` is described as committed but is not tracked by git"))
        elif claim in ("ignored", "untracked") and is_tracked:
            problems.append((n, f"`{path}` is described as {'ignored' if claim == 'ignored' else 'not committed'} but is tracked by git"))
        elif claim == "ignored" and not git_ignored(path, root):
            problems.append((n, f"`{path}` is described as ignored but .gitignore does not ignore it"))
    return problems


def check(paths: list[Path], root: Path = ROOT) -> list[str]:
    problems = []
    tracked = {str(p.relative_to(root)) for p in paths if p.is_relative_to(root)}
    for path in paths:
        text = read_text(path)
        if text is None:
            continue
        found = conflict_markers(text)
        if path.suffix.lower() == ".md":
            found += duplicated_markdown(text) + claim_problems(text, tracked, root)
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        problems += [f"{rel}:{n}: {message}" for n, message in sorted(found)]
    return problems


def main() -> int:
    problems = check(tracked_files())
    print("\n".join(problems) if problems else "repository checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
