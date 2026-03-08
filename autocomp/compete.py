"""
AutoComp utilities — helpers for competition workflow.

This module provides:
- Competition setup (download data, initialize directories)
- CV evaluation framework
- Experiment tracking (append to journal)
- Kaggle API wrappers (submit, check score)
- Cross-competition memory search
"""

import os
import re
import json
import shutil
import subprocess
import datetime
import yaml
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AUTOCOMP_ROOT = Path(__file__).parent
COMPETITIONS_DIR = AUTOCOMP_ROOT / "competitions"
TEMPLATES_DIR = AUTOCOMP_ROOT / "templates"

GITIGNORE_PATTERNS = [
    "data/",
    "artifacts/models/",
    "artifacts/oof/",
    "submissions/",
    "__pycache__/",
    "*.pyc",
]


# ---------------------------------------------------------------------------
# Competition setup
# ---------------------------------------------------------------------------


def init_competition(slug: str, name: Optional[str] = None) -> Path:
    """Create the directory structure for a new competition.

    Args:
        slug: Kaggle competition slug (e.g. 'titanic')
        name: Human-readable name (defaults to slug)

    Returns:
        Path to the competition directory.
    """
    name = name or slug.replace("-", " ").title()
    comp_dir = COMPETITIONS_DIR / slug

    if comp_dir.exists():
        print(f"Competition directory already exists: {comp_dir}")
        return comp_dir

    # Create directory tree
    dirs = [
        "data",
        "notebooks",
        "submissions",
        "artifacts/models",
        "artifacts/oof",
        "artifacts/plots",
        "artifacts/feature_importance",
    ]
    for d in dirs:
        (comp_dir / d).mkdir(parents=True, exist_ok=True)

    # Copy and fill journal template
    journal_template = (TEMPLATES_DIR / "journal.md").read_text()
    journal = journal_template.replace("{competition_name}", name)
    journal = journal.replace("{kaggle_url}", f"https://www.kaggle.com/competitions/{slug}")
    # Leave other placeholders for the agent to fill in during Research phase
    (comp_dir / "journal.md").write_text(journal)

    # Copy config template
    config_template = (TEMPLATES_DIR / "config.yaml").read_text()
    config = yaml.safe_load(config_template)
    config["competition"]["name"] = name
    config["competition"]["slug"] = slug
    config["competition"]["url"] = f"https://www.kaggle.com/competitions/{slug}"
    with open(comp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Create .gitignore
    (comp_dir / ".gitignore").write_text("\n".join(GITIGNORE_PATTERNS) + "\n")

    print(f"Initialized competition: {comp_dir}")
    return comp_dir


def download_data(slug: str) -> None:
    """Download competition data using the Kaggle CLI.

    Args:
        slug: Kaggle competition slug.
    """
    comp_dir = COMPETITIONS_DIR / slug
    data_dir = comp_dir / "data"

    if not comp_dir.exists():
        raise FileNotFoundError(f"Run init_competition('{slug}') first.")

    print(f"Downloading data for {slug}...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", slug, "-p", str(data_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error downloading data:\n{result.stderr}")
        return

    # Unzip if a zip file was downloaded
    for zf in data_dir.glob("*.zip"):
        print(f"Unzipping {zf.name}...")
        subprocess.run(["unzip", "-o", str(zf), "-d", str(data_dir)], capture_output=True)
        zf.unlink()

    print(f"Data downloaded to {data_dir}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(slug: str) -> dict:
    """Load the competition config."""
    config_path = COMPETITIONS_DIR / slug / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(slug: str, config: dict) -> None:
    """Save the competition config."""
    config_path = COMPETITIONS_DIR / slug / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def update_config(slug: str, **kwargs) -> None:
    """Update specific fields in the competition config.

    Supports dotted keys: update_config('titanic', status__best_cv=0.82)
    """
    config = load_config(slug)
    for key, value in kwargs.items():
        parts = key.split("__")
        d = config
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = value
    save_config(slug, config)


# ---------------------------------------------------------------------------
# Journal helpers
# ---------------------------------------------------------------------------


def read_journal(slug: str) -> str:
    """Read the competition journal."""
    journal_path = COMPETITIONS_DIR / slug / "journal.md"
    return journal_path.read_text()


def append_session(slug: str, session_num: int, date: str, phase: str,
                   goal: str, did: list[str], learned: list[str],
                   results: list[str], next_session: list[str]) -> None:
    """Append a new session entry to the journal.

    This inserts the session entry just before the Experiment Log section.

    Args:
        slug: Competition slug.
        session_num: Session number.
        date: Date string (YYYY-MM-DD).
        phase: Phase name (research, eda, baseline, etc.)
        goal: What this session aimed to do.
        did: List of actions taken.
        learned: List of insights.
        results: List of measurable outcomes.
        next_session: List of next steps.
    """
    journal_path = COMPETITIONS_DIR / slug / "journal.md"
    content = journal_path.read_text()

    entry = f"\n### Session {session_num} — {date} — {phase.title()}\n"
    entry += f"**Goal:** {goal}\n\n"
    entry += "**What I did:**\n" + "".join(f"- {x}\n" for x in did) + "\n"
    entry += "**What I learned:**\n" + "".join(f"- {x}\n" for x in learned) + "\n"
    entry += "**Results:**\n" + "".join(f"- {x}\n" for x in results) + "\n"
    entry += "**Next session should:**\n" + "".join(f"- {x}\n" for x in next_session) + "\n"
    entry += "---\n"

    # Insert before Experiment Log
    marker = "## Experiment Log"
    if marker in content:
        content = content.replace(marker, entry + "\n" + marker)
    else:
        content += "\n" + entry

    journal_path.write_text(content)
    print(f"Session {session_num} appended to journal.")


def append_experiment(slug: str, num: int, session: int, phase: str,
                      change: str, hypothesis: str, cv_score: float,
                      lb_score: Optional[float], delta: float, keep: bool) -> None:
    """Append a row to the experiment log table in the journal."""
    journal_path = COMPETITIONS_DIR / slug / "journal.md"
    content = journal_path.read_text()

    lb_str = f"{lb_score:.4f}" if lb_score is not None else "—"
    delta_str = f"{delta:+.4f}" if delta != 0 else "—"
    keep_str = "✓" if keep else "✗"

    row = f"| {num} | {session} | {phase} | {change} | {hypothesis} | {cv_score:.4f} | {lb_str} | {delta_str} | {keep_str} |\n"

    # Find the end of the experiment log table and append
    lines = content.split("\n")
    table_end = None
    in_table = False
    for i, line in enumerate(lines):
        if "## Experiment Log" in line:
            in_table = True
        elif in_table and line.startswith("|"):
            table_end = i
        elif in_table and not line.startswith("|") and line.strip() == "":
            if table_end is not None:
                break

    if table_end is not None:
        lines.insert(table_end + 1, row.rstrip())
        content = "\n".join(lines)
    else:
        # Fallback: append after the header row
        content = content.replace(
            "|---|---------|-------|--------|------------|----------|----------|-------|------|",
            "|---|---------|-------|--------|------------|----------|----------|-------|------|\n" + row.rstrip(),
        )

    journal_path.write_text(content)


# ---------------------------------------------------------------------------
# Kaggle API wrappers
# ---------------------------------------------------------------------------


def submit(slug: str, filepath: str, message: str = "") -> Optional[str]:
    """Submit a prediction file to Kaggle.

    Args:
        slug: Competition slug.
        filepath: Path to the submission CSV.
        message: Description of this submission.

    Returns:
        Submission output or None on error.
    """
    if not message:
        message = f"AutoComp submission {datetime.datetime.now().isoformat()}"

    result = subprocess.run(
        ["kaggle", "competitions", "submit", "-c", slug, "-f", filepath, "-m", message],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Submission error:\n{result.stderr}")
        return None

    print(f"Submitted: {filepath}")
    print(result.stdout)
    return result.stdout


def get_leaderboard_score(slug: str) -> Optional[float]:
    """Get the most recent submission score from Kaggle.

    Returns:
        The public leaderboard score, or None if unavailable.
    """
    result = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", slug, "--csv"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error fetching submissions:\n{result.stderr}")
        return None

    lines = result.stdout.strip().split("\n")
    if len(lines) < 2:
        return None

    # Parse CSV — score is typically the 'publicScore' column
    import csv
    import io
    reader = csv.DictReader(io.StringIO(result.stdout))
    for row in reader:
        score_str = row.get("publicScore", "")
        try:
            return float(score_str)
        except (ValueError, TypeError):
            return None

    return None


# ---------------------------------------------------------------------------
# Cross-competition memory
# ---------------------------------------------------------------------------


def search_past_learnings(query: str) -> list[dict]:
    """Search past competition journals for relevant learnings.

    Looks at the 'Learnings (Transferable)' section of each past journal.

    Args:
        query: Search terms (e.g., 'tabular classification imbalanced')

    Returns:
        List of {slug, learnings} dicts with matching content.
    """
    results = []
    query_terms = query.lower().split()

    for comp_dir in COMPETITIONS_DIR.iterdir():
        if not comp_dir.is_dir():
            continue
        journal_path = comp_dir / "journal.md"
        if not journal_path.exists():
            continue

        content = journal_path.read_text()

        # Extract learnings section
        learnings_match = re.search(
            r"## Learnings \(Transferable\)\s*\n(.*?)(?=\n## |\Z)",
            content,
            re.DOTALL,
        )
        if not learnings_match:
            continue

        learnings = learnings_match.group(1).strip()
        if not learnings or learnings.startswith("{"):
            continue  # Skip unfilled templates

        # Simple keyword matching
        learnings_lower = learnings.lower()
        if any(term in learnings_lower for term in query_terms):
            results.append({
                "slug": comp_dir.name,
                "learnings": learnings,
            })

    return results


def list_competitions() -> list[dict]:
    """List all competition directories with their status.

    Returns:
        List of {slug, name, phase, best_cv, best_lb} dicts.
    """
    comps = []
    for comp_dir in sorted(COMPETITIONS_DIR.iterdir()):
        if not comp_dir.is_dir():
            continue
        config_path = comp_dir / "config.yaml"
        if not config_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        comps.append({
            "slug": comp_dir.name,
            "name": config.get("competition", {}).get("name", comp_dir.name),
            "phase": config.get("status", {}).get("current_phase", "unknown"),
            "best_cv": config.get("status", {}).get("best_cv"),
            "best_lb": config.get("status", {}).get("best_lb"),
        })
    return comps


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Simple CLI for common operations."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python compete.py init <slug> [name]    Initialize a competition")
        print("  python compete.py download <slug>        Download competition data")
        print("  python compete.py list                   List all competitions")
        print("  python compete.py search <query>         Search past learnings")
        return

    cmd = sys.argv[1]

    if cmd == "init":
        slug = sys.argv[2]
        name = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else None
        init_competition(slug, name)

    elif cmd == "download":
        slug = sys.argv[2]
        download_data(slug)

    elif cmd == "list":
        for comp in list_competitions():
            status = f"[{comp['phase']}]"
            cv = f"CV={comp['best_cv']}" if comp['best_cv'] else "no score"
            lb = f"LB={comp['best_lb']}" if comp['best_lb'] else ""
            print(f"  {comp['slug']:30s} {status:15s} {cv} {lb}")

    elif cmd == "search":
        query = " ".join(sys.argv[2:])
        results = search_past_learnings(query)
        if not results:
            print("No matching learnings found.")
        for r in results:
            print(f"\n--- {r['slug']} ---")
            print(r["learnings"])

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
