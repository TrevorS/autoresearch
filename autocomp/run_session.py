"""
run_session.py — Runs inside the Docker container.

Uses the Anthropic Python SDK to run an agent loop:
  prompt → Claude → tool calls → execute → feed results → repeat

The orchestrator (on the host) manages container lifecycle.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import anthropic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AUTOCOMP_DIR = Path(__file__).parent
COMPETITIONS_DIR = AUTOCOMP_DIR / "competitions"

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 16384

# Tool definitions for the Anthropic API
TOOLS = [
    {
        "name": "bash",
        "description": (
            "Execute a bash command. Use for running scripts, git, kaggle CLI, "
            "pip, data processing, etc. Commands run in the competition directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file's contents. Use absolute paths or paths relative to the competition directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory, optionally with a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob pattern (e.g. '*.csv', '**/*.py').",
                },
            },
            "required": ["path"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool execution (sandboxed inside Docker)
# ---------------------------------------------------------------------------


def execute_tool(name: str, input: dict, cwd: Path) -> str:
    """Execute a tool call and return the result as a string."""

    if name == "bash":
        return _exec_bash(input["command"], cwd)
    elif name == "read_file":
        return _exec_read(input["path"], cwd)
    elif name == "write_file":
        return _exec_write(input["path"], input["content"], cwd)
    elif name == "list_files":
        return _exec_list(input["path"], input.get("pattern"), cwd)
    else:
        return f"Unknown tool: {name}"


def _exec_bash(command: str, cwd: Path) -> str:
    """Execute a bash command with a timeout."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(cwd),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + f"[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "[error: command timed out after 600s]"
    except Exception as e:
        return f"[error: {e}]"


def _exec_read(path: str, cwd: Path) -> str:
    """Read a file."""
    try:
        p = Path(path)
        if not p.is_absolute():
            p = cwd / p
        content = p.read_text()
        if len(content) > 100000:
            return content[:100000] + f"\n\n[truncated — file is {len(content)} chars]"
        return content
    except Exception as e:
        return f"[error reading {path}: {e}]"


def _exec_write(path: str, content: str, cwd: Path) -> str:
    """Write a file."""
    try:
        p = Path(path)
        if not p.is_absolute():
            p = cwd / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {p}"
    except Exception as e:
        return f"[error writing {path}: {e}]"


def _exec_list(path: str, pattern: str | None, cwd: Path) -> str:
    """List files in a directory."""
    try:
        p = Path(path)
        if not p.is_absolute():
            p = cwd / p
        if pattern:
            files = sorted(p.glob(pattern))
        else:
            files = sorted(p.iterdir())
        lines = []
        for f in files[:500]:
            suffix = "/" if f.is_dir() else ""
            try:
                size = f.stat().st_size if f.is_file() else 0
                lines.append(f"{f.relative_to(p)}{suffix}  ({size} bytes)" if size else f"{f.relative_to(p)}{suffix}")
            except OSError:
                lines.append(str(f.relative_to(p)) + suffix)
        if len(files) > 500:
            lines.append(f"... and {len(files) - 500} more files")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as e:
        return f"[error listing {path}: {e}]"


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def load_program() -> str:
    return (AUTOCOMP_DIR / "program.md").read_text()


def load_journal(slug: str) -> str:
    journal_path = COMPETITIONS_DIR / slug / "journal.md"
    return journal_path.read_text() if journal_path.exists() else ""


def load_config(slug: str) -> dict:
    import yaml
    config_path = COMPETITIONS_DIR / slug / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def gather_past_learnings(slug: str) -> str:
    learnings = []
    if not COMPETITIONS_DIR.exists():
        return ""
    for comp_dir in COMPETITIONS_DIR.iterdir():
        if not comp_dir.is_dir() or comp_dir.name == slug:
            continue
        journal_path = comp_dir / "journal.md"
        if not journal_path.exists():
            continue
        content = journal_path.read_text()
        if "## Learnings (Transferable)" in content:
            idx = content.index("## Learnings (Transferable)")
            section = content[idx:]
            next_section = section.find("\n## ", 3)
            if next_section != -1:
                section = section[:next_section]
            if "{" not in section[:100]:  # skip unfilled templates
                learnings.append(f"### From: {comp_dir.name}\n{section}")
    return "\n\n".join(learnings)


def build_system_prompt(slug: str) -> str:
    config = load_config(slug)
    comp_dir = COMPETITIONS_DIR / slug

    parts = [
        "You are an autonomous Kaggle competition agent.",
        f"Competition: {config.get('competition', {}).get('name', slug)}",
        f"Slug: {slug}",
        f"Working directory: {comp_dir}",
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
        "You have these tools: bash, read_file, write_file, list_files.",
        "All file paths are relative to the working directory unless absolute.",
        "",
        "## Instructions",
        "",
        load_program(),
    ]

    past = gather_past_learnings(slug)
    if past:
        parts.extend(["", "## Learnings from Past Competitions", "", past])

    return "\n".join(parts)


def build_user_prompt(slug: str) -> str:
    journal = load_journal(slug)
    comp_dir = COMPETITIONS_DIR / slug

    parts = [
        "## Current Journal",
        "",
        journal if journal else "*No journal yet — this is your first session. Start with Phase 1 (Research).*",
        "",
        "---",
        "",
        "## Your Task This Session",
        "",
        "Read the journal above. The 'Next session should' section tells you what to do.",
        "If there's no journal yet, start with Phase 1 (Research).",
        "",
        "When you're done:",
        "1. Update the journal with a new session entry (what you did, learned, results, next steps)",
        "2. Update config.yaml if scores or phase changed",
        "3. Commit all changes with a descriptive message",
        "",
        f"All work goes in `{comp_dir}/`. Do not modify files outside this directory.",
    ]

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_session(slug: str, max_turns: int = 100, max_budget_usd: float = 5.0) -> dict:
    """Run a single agent session.

    Implements the tool-use loop:
      1. Send messages to Claude
      2. If Claude responds with tool_use blocks, execute them
      3. Feed tool results back
      4. Repeat until Claude stops calling tools or limits hit
    """
    comp_dir = COMPETITIONS_DIR / slug
    client = anthropic.Anthropic()

    system_prompt = build_system_prompt(slug)
    messages = [{"role": "user", "content": build_user_prompt(slug)}]

    result = {
        "slug": slug,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "num_turns": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "duration_seconds": 0,
        "tool_calls": [],
    }

    start_time = time.time()
    print(f"[autocomp] Starting session for '{slug}'")
    print(f"[autocomp] Max turns: {max_turns}")
    print("=" * 60)

    for turn in range(max_turns):
        result["num_turns"] = turn + 1

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            print(f"[autocomp] API error: {e}")
            result["status"] = "api_error"
            result["error"] = str(e)
            break

        # Track token usage
        result["total_input_tokens"] += response.usage.input_tokens
        result["total_output_tokens"] += response.usage.output_tokens

        # Process response blocks
        assistant_content = response.content
        tool_uses = []

        for block in assistant_content:
            if block.type == "text" and block.text.strip():
                text = block.text.strip()
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"[agent] {preview}")
            elif block.type == "tool_use":
                tool_uses.append(block)
                result["tool_calls"].append(block.name)
                if block.name == "bash":
                    print(f"[tool] bash: {block.input.get('command', '')[:100]}")
                elif block.name == "write_file":
                    print(f"[tool] write_file: {block.input.get('path', '')}")
                elif block.name == "read_file":
                    print(f"[tool] read_file: {block.input.get('path', '')}")
                else:
                    print(f"[tool] {block.name}")

        # Append assistant message to conversation
        messages.append({"role": "assistant", "content": assistant_content})

        # If no tool calls, the agent is done
        if not tool_uses:
            print("[autocomp] Agent finished (no more tool calls).")
            break

        # Execute tool calls and build tool results
        tool_results = []
        for tool_use in tool_uses:
            output = execute_tool(tool_use.name, tool_use.input, comp_dir)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": output,
            })

        messages.append({"role": "user", "content": tool_results})

        # Check stop reason
        if response.stop_reason == "end_turn":
            # Claude said it's done but also called tools — continue after results
            pass

    else:
        print(f"[autocomp] Hit max turns ({max_turns}).")
        result["status"] = "max_turns"

    result["duration_seconds"] = round(time.time() - start_time, 1)

    print("=" * 60)
    print(f"[autocomp] Status: {result['status']}")
    print(f"[autocomp] Turns: {result['num_turns']}")
    print(f"[autocomp] Tokens: {result['total_input_tokens']} in / {result['total_output_tokens']} out")
    print(f"[autocomp] Duration: {result['duration_seconds']}s")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run an autocomp agent session")
    parser.add_argument("slug", help="Kaggle competition slug")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-budget", type=float, default=5.0)
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model

    comp_dir = COMPETITIONS_DIR / args.slug
    if not comp_dir.exists():
        print(f"Competition not found: {comp_dir}")
        print(f"Run: python compete.py init {args.slug}")
        sys.exit(1)

    result = run_session(args.slug, args.max_turns, args.max_budget)

    # Write result for orchestrator
    log_path = comp_dir / "last_session.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[autocomp] Result written to {log_path}")

    if result["status"] not in ("success", "max_turns"):
        sys.exit(1)


if __name__ == "__main__":
    main()
