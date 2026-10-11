"""Build a self-contained, non-data submission ZIP."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "submission"
NAME = "Business_Analytics_Finance_Team"
STAGE = OUT / NAME
ZIP = OUT / f"{NAME}.zip"
EXCLUDED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", "cache", "dataset", "output", "reports", "tmp", "wandb"}


def allowed_python(path: Path) -> bool:
    return path.suffix == ".py" and not any(part in EXCLUDED_PARTS for part in path.parts)


def copy_file(src: Path, rel: Path) -> None:
    dst = STAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    # Deletion targets are fixed children of output/submission.
    OUT.mkdir(parents=True, exist_ok=True)
    if STAGE.exists():
        shutil.rmtree(STAGE)
    if ZIP.exists():
        ZIP.unlink()
    STAGE.mkdir()
    for filename in ("README.md", "main.py", "pyproject.toml", "uv.lock"):
        copy_file(ROOT / filename, Path(filename))
    for src in ROOT.rglob("*.py"):
        if allowed_python(src):
            copy_file(src, src.relative_to(ROOT))
    copy_file(ROOT / "lab" / "folds.json", Path("lab/folds.json"))
    for name in ("gapz_rule.json", "premarket_rule.json", "premarket_rule.provenance.json"):
        copy_file(ROOT / "artifacts" / name, Path("artifacts") / name)
    # Ship the store the frozen rule actually reads; the other cutoff's store would be dead weight.
    as_of = json.loads((ROOT / "artifacts" / "premarket_rule.json").read_text(encoding="utf-8"))["as_of"]
    store = "premarket" if as_of == "09:15" else f"premarket_{as_of.replace(':', '')}"
    files = sorted((ROOT / "artifacts" / store).glob("*.parquet"))
    assert files, f"no files in artifacts/{store}"
    for src in files:
        copy_file(src, Path("artifacts") / store / src.name)
    for rel in ("reports/premarket_submission_check.json", "reports/premarket_holdout/result.json",
                "reports/premarket_compare/0929_vs_0928.json", "reports/premarket_compare/bootstrap.json"):
        copy_file(ROOT / rel, Path(rel))
    copy_file(ROOT / "output" / "pdf" / "Finance_Final_Presentation.pdf", Path("Presentation.pdf"))
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in sorted(STAGE.rglob("*")):
            if src.is_file():
                zf.write(src, src.relative_to(OUT))
    print(f"stage={STAGE}")
    print(f"zip={ZIP}")
    print(f"files={sum(1 for p in STAGE.rglob('*') if p.is_file())}")


if __name__ == "__main__":
    main()
