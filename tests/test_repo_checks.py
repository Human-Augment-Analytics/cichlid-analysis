"""Regression tests for scripts/check_repo.py and for doc claims checked against code.

Each test guards a mistake that has reached main before: merge-conflict markers
left in docs, uv.lock and a marimo notebook; sections pasted twice in a README;
README statements that contradicted git or the CLI defaults.
"""
import ast
import importlib.util
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_repo", ROOT / "scripts" / "check_repo.py")
check_repo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_repo)


def test_repository_passes_all_checks():
    assert check_repo.check(check_repo.tracked_files()) == []


def test_conflict_markers_are_found():
    text = "a\n<<<<<<< HEAD\nb\n=======\nc\n>>>>>>> b27b994 (added all new models)\nd\n"
    assert [n for n, _ in check_repo.conflict_markers(text)] == [2, 4, 6]


def test_conflict_marker_lookalikes_are_ignored():
    text = "Title\n=======\n\n<<<<<<<<<< not a marker\n=== short ===\n"
    assert check_repo.conflict_markers(text) == []


def test_lines_pasted_twice_are_found():
    para = ("Joint metrics (`direction_f1`, cycle MAE at IoU >= 0.1) are the end-to-end\n"
            "score (Layer C). Use the strict-IoU and oracle-span tables to judge quality.\n")
    text = f"# Doc\n\n{para}\nOther text.\n\nwhich is then evaluated as its own model. For that estimator\n{para}"
    assert [n for n, _ in check_repo.duplicated_markdown(text)] == [9, 10]


def test_adjacent_repeat_inside_code_fence_is_found():
    line = "from evaluation.evaluate import (evaluate, prediction_models, prediction_video_ids,"
    text = f"```python\n{line}\n{line}\n```\n"
    assert [n for n, _ in check_repo.duplicated_markdown(text)] == [3]


def test_short_lines_tables_and_separate_code_repeats_are_allowed():
    arg = "  --annotations ground_truth_annotations/finalized_annotations/circling.csv \\"
    text = (f"```sh\nrun a \\\n{arg}\n```\n\n```sh\nrun b \\\n{arg}\n```\n\n"
            "| a | b |\n|---|---|\n|---|---|\n\n- `video_id`\n- `video_id`\n")
    assert check_repo.duplicated_markdown(text) == []


def test_committed_and_ignored_claims_are_checked_against_git():
    tracked = {"feature_engineering/pair_motion/data/sample_pair_features.csv", "outputs/predictions/a/b.csv"}
    text = ("- `feature_engineering/pair_motion/data/sample_pair_features.csv`: small, committed sample\n"
            "- `outputs/predictions/`: committed predictions\n"
            "- `outputs/pair_motion/pair_features_frame.parquet`: full feature table; ignored.\n"
            "- `outputs/never_made.csv`: committed copy\n"
            "- `outputs/predictions/`: generated, ignored\n"
            "- `models/new_thing/`: scratch files, ignored\n")
    problems = check_repo.claim_problems(text, tracked)
    assert [n for n, _ in problems] == [4, 5, 6]
    assert "not tracked" in problems[0][1]
    assert "is tracked" in problems[1][1]
    assert ".gitignore" in problems[2][1]


def argparse_default(path, flag):
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and getattr(node.args[0], "value", None) == flag):
            return next(ast.literal_eval(k.value) for k in node.keywords if k.arg == "default")
    raise AssertionError(f"{flag} not found in {path}")


def test_readme_mstcn_default_features_match_cli():
    match = re.search(r"### MS-TCN\n+The default feature set is `([^`]+)`", (ROOT / "README.md").read_text())
    assert match, "README MS-TCN section no longer states the default feature set"
    assert match.group(1) == argparse_default(ROOT / "models/intervals/mstcn_dlc2action/train.py", "--features")


def test_pair_motion_sample_size_matches_docs():
    doc = (ROOT / "feature_engineering/pair_motion/README.md").read_text()
    rows = int(re.search(r"up to (\d+) rows", doc).group(1))
    tree = ast.parse((ROOT / "feature_engineering/pair_motion/build_features.py").read_text())
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "write_sample")
    defaults = dict(zip([a.arg for a in func.args.args][-len(func.args.defaults):], func.args.defaults))
    assert ast.literal_eval(defaults["rows_per_video"]) == rows


def test_lockfile_has_no_conflict_markers_and_matches_pyproject():
    assert check_repo.conflict_markers((ROOT / "uv.lock").read_text()) == []
    result = subprocess.run(["uv", "lock", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
