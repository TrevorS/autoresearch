"""
orchestrator.py — Runs on the host machine.

Manages the competition lifecycle by:
1. Building the Docker image (once)
2. Running agent sessions in containers
3. Reading session results and deciding what to do next
4. Logging everything to a session history

Usage:
    # Run a single session
    python orchestrator.py run titanic

    # Run sessions in a loop until the agent says it's done or budget exhausted
    python orchestrator.py loop titanic --max-sessions 20 --session-budget 5.0

    # Initialize a new competition
    python orchestrator.py init titanic "Titanic Survival Prediction"

    # Check status
    python orchestrator.py status titanic
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

AUTOCOMP_DIR = Path(__file__).parent
COMPETITIONS_DIR = AUTOCOMP_DIR / "competitions"
IMAGE_NAME = "autocomp-agent"


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def build_image(force: bool = False) -> bool:
    """Build the Docker image for the agent environment."""
    # Check if image already exists
    if not force:
        result = subprocess.run(
            ["docker", "image", "inspect", IMAGE_NAME],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"[orchestrator] Image '{IMAGE_NAME}' already exists (use --rebuild to force)")
            return True

    print(f"[orchestrator] Building image '{IMAGE_NAME}'...")
    result = subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, "-f", str(AUTOCOMP_DIR / "Dockerfile"), str(AUTOCOMP_DIR.parent)],
        capture_output=False,  # stream build output
    )
    if result.returncode != 0:
        print("[orchestrator] Docker build failed!")
        return False

    print(f"[orchestrator] Image built successfully.")
    return True


def run_container(
    slug: str,
    max_turns: int = 100,
    max_budget: float = 5.0,
    model: str | None = None,
    timeout: int = 3600,
) -> dict:
    """Run an agent session in a Docker container.

    Args:
        slug: Competition slug.
        max_turns: Max agent turns per session.
        max_budget: Max cost per session in USD.
        model: Override the Claude model.
        timeout: Container timeout in seconds (default 1 hour).

    Returns:
        Session result dict, or error dict on failure.
    """
    comp_dir = COMPETITIONS_DIR / slug
    if not comp_dir.exists():
        return {"status": "error", "error": f"Competition not found: {comp_dir}"}

    # Build docker run command
    cmd = [
        "docker", "run",
        "--rm",                                         # cleanup after exit
        "--name", f"autocomp-{slug}-{int(time.time())}",
        # Mount competition workspace (read-write)
        "-v", f"{comp_dir.resolve()}:/workspace/autocomp/competitions/{slug}",
        # Mount autocomp code (read-only, so agent can read program.md)
        "-v", f"{AUTOCOMP_DIR.resolve()}/program.md:/workspace/autocomp/program.md:ro",
        "-v", f"{AUTOCOMP_DIR.resolve()}/compete.py:/workspace/autocomp/compete.py:ro",
        "-v", f"{AUTOCOMP_DIR.resolve()}/templates:/workspace/autocomp/templates:ro",
        # Mount other competition dirs read-only (for cross-competition learning)
        "-v", f"{COMPETITIONS_DIR.resolve()}:/workspace/autocomp/competitions:ro",
        # Re-mount THIS competition read-write (overrides the ro mount above)
        "-v", f"{comp_dir.resolve()}:/workspace/autocomp/competitions/{slug}",
        # Pass API key
        "-e", "ANTHROPIC_API_KEY",
        # Mount Kaggle credentials if available
    ]

    # Mount kaggle credentials if they exist
    kaggle_dir = Path.home() / ".kaggle"
    if kaggle_dir.exists():
        cmd.extend(["-v", f"{kaggle_dir}:/root/.kaggle:ro"])

    # Network access for API calls and data downloads
    cmd.extend(["--network", "bridge"])

    # Resource limits
    cmd.extend([
        "--memory", "8g",
        "--cpus", "4",
    ])

    # Image and entrypoint args
    cmd.append(IMAGE_NAME)
    cmd.extend([slug, "--max-turns", str(max_turns), "--max-budget", str(max_budget)])
    if model:
        cmd.extend(["--model", model])

    print(f"[orchestrator] Starting container for '{slug}'...")
    print(f"[orchestrator] Max turns: {max_turns}, Budget: ${max_budget:.2f}, Timeout: {timeout}s")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=False,  # stream container output to terminal
        )
    except subprocess.TimeoutExpired:
        print(f"[orchestrator] Container timed out after {timeout}s!")
        # Kill the container
        subprocess.run(
            ["docker", "kill", f"autocomp-{slug}-{int(start_time)}"],
            capture_output=True,
        )
        return {"status": "timeout", "duration_seconds": timeout}

    duration = round(time.time() - start_time, 1)

    # Read the session result from the competition directory
    result_path = comp_dir / "last_session.json"
    if result_path.exists():
        session_result = json.loads(result_path.read_text())
        session_result["container_duration_seconds"] = duration
        return session_result
    else:
        return {
            "status": "error" if result.returncode != 0 else "unknown",
            "container_exit_code": result.returncode,
            "container_duration_seconds": duration,
        }


# ---------------------------------------------------------------------------
# Session loop
# ---------------------------------------------------------------------------


def run_loop(
    slug: str,
    max_sessions: int = 20,
    session_budget: float = 5.0,
    max_turns_per_session: int = 100,
    total_budget: float = 50.0,
    cooldown: int = 10,
    model: str | None = None,
) -> list[dict]:
    """Run multiple agent sessions in a loop.

    Stops when:
    - max_sessions reached
    - total_budget exhausted
    - Agent signals completion (Phase 7 done)
    - Consecutive errors

    Args:
        slug: Competition slug.
        max_sessions: Maximum number of sessions to run.
        session_budget: Budget per session in USD.
        max_turns_per_session: Max turns per session.
        total_budget: Total budget across all sessions.
        cooldown: Seconds to wait between sessions.
        model: Override model name.

    Returns:
        List of session results.
    """
    results = []
    total_cost = 0.0
    consecutive_errors = 0

    print(f"[orchestrator] Starting loop for '{slug}'")
    print(f"[orchestrator] Max sessions: {max_sessions}, Total budget: ${total_budget:.2f}")
    print()

    for i in range(max_sessions):
        print(f"\n{'='*60}")
        print(f"[orchestrator] Session {i+1}/{max_sessions} | Total cost so far: ${total_cost:.4f}")
        print(f"{'='*60}\n")

        session_result = run_container(
            slug,
            max_turns=max_turns_per_session,
            max_budget=min(session_budget, total_budget - total_cost),
            model=model,
        )
        results.append(session_result)

        # Track cost (estimate from tokens if not provided)
        session_cost = session_result.get("cost_usd", 0.0)
        total_cost += session_cost

        # Check for errors
        if session_result.get("status") in ("error", "api_error", "timeout"):
            consecutive_errors += 1
            print(f"[orchestrator] Session failed: {session_result.get('status')}")
            if consecutive_errors >= 3:
                print("[orchestrator] 3 consecutive errors — stopping.")
                break
        else:
            consecutive_errors = 0

        # Check budget
        if total_cost >= total_budget:
            print(f"[orchestrator] Total budget exhausted (${total_cost:.4f} >= ${total_budget:.2f})")
            break

        # Check if competition is done (Phase 7 complete)
        config = _load_config(slug)
        if config.get("status", {}).get("current_phase") == "final_done":
            print("[orchestrator] Competition marked as complete!")
            break

        # Cooldown between sessions
        if i < max_sessions - 1:
            print(f"[orchestrator] Cooling down {cooldown}s before next session...")
            time.sleep(cooldown)

    # Write session history
    history_path = COMPETITIONS_DIR / slug / "session_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "slug": slug,
            "total_sessions": len(results),
            "total_cost_usd": total_cost,
            "sessions": results,
        }, f, indent=2)

    print(f"\n[orchestrator] Loop complete: {len(results)} sessions, ${total_cost:.4f} total")
    print(f"[orchestrator] History saved to {history_path}")

    return results


def _load_config(slug: str) -> dict:
    import yaml
    config_path = COMPETITIONS_DIR / slug / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# ---------------------------------------------------------------------------
# Init helper (delegates to compete.py)
# ---------------------------------------------------------------------------


def init_competition(slug: str, name: str | None = None):
    """Initialize a competition (runs compete.py locally)."""
    sys.path.insert(0, str(AUTOCOMP_DIR))
    from compete import init_competition as _init, download_data
    _init(slug, name)
    print(f"\nTo download data, run: python orchestrator.py download {slug}")


def download_data(slug: str):
    """Download competition data."""
    sys.path.insert(0, str(AUTOCOMP_DIR))
    from compete import download_data as _download
    _download(slug)


def show_status(slug: str):
    """Show competition status from config and last session."""
    config = _load_config(slug)
    comp = config.get("competition", {})
    status = config.get("status", {})

    print(f"Competition: {comp.get('name', slug)}")
    print(f"Phase:       {status.get('current_phase', 'unknown')}")
    print(f"Best CV:     {status.get('best_cv', 'n/a')}")
    print(f"Best LB:     {status.get('best_lb', 'n/a')}")
    print(f"Experiments: {status.get('total_experiments', 0)}")
    print(f"Submissions: {status.get('total_submissions', 0)}")

    # Show last session info
    last_session = COMPETITIONS_DIR / slug / "last_session.json"
    if last_session.exists():
        session = json.loads(last_session.read_text())
        print(f"\nLast session:")
        print(f"  Status:   {session.get('status', 'unknown')}")
        print(f"  Turns:    {session.get('num_turns', 'unknown')}")
        print(f"  Duration: {session.get('duration_seconds', 'unknown')}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AutoComp orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialize a competition")
    p_init.add_argument("slug")
    p_init.add_argument("name", nargs="?")

    # download
    p_dl = subparsers.add_parser("download", help="Download competition data")
    p_dl.add_argument("slug")

    # build
    p_build = subparsers.add_parser("build", help="Build the Docker image")
    p_build.add_argument("--rebuild", action="store_true")

    # run (single session)
    p_run = subparsers.add_parser("run", help="Run a single agent session")
    p_run.add_argument("slug")
    p_run.add_argument("--max-turns", type=int, default=100)
    p_run.add_argument("--max-budget", type=float, default=5.0)
    p_run.add_argument("--model", type=str, default=None)
    p_run.add_argument("--timeout", type=int, default=3600)

    # loop (multiple sessions)
    p_loop = subparsers.add_parser("loop", help="Run sessions in a loop")
    p_loop.add_argument("slug")
    p_loop.add_argument("--max-sessions", type=int, default=20)
    p_loop.add_argument("--session-budget", type=float, default=5.0)
    p_loop.add_argument("--total-budget", type=float, default=50.0)
    p_loop.add_argument("--max-turns", type=int, default=100)
    p_loop.add_argument("--cooldown", type=int, default=10)
    p_loop.add_argument("--model", type=str, default=None)

    # status
    p_status = subparsers.add_parser("status", help="Show competition status")
    p_status.add_argument("slug")

    args = parser.parse_args()

    if args.command == "init":
        init_competition(args.slug, args.name)

    elif args.command == "download":
        download_data(args.slug)

    elif args.command == "build":
        build_image(force=args.rebuild)

    elif args.command == "run":
        if not build_image():
            sys.exit(1)
        result = run_container(args.slug, args.max_turns, args.max_budget, args.model, args.timeout)
        print(json.dumps(result, indent=2))

    elif args.command == "loop":
        if not build_image():
            sys.exit(1)
        run_loop(
            args.slug,
            max_sessions=args.max_sessions,
            session_budget=args.session_budget,
            max_turns_per_session=args.max_turns,
            total_budget=args.total_budget,
            cooldown=args.cooldown,
            model=args.model,
        )

    elif args.command == "status":
        show_status(args.slug)


if __name__ == "__main__":
    main()
