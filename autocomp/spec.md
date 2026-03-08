# autocomp spec

Autonomous agent that competes in Kaggle competitions. Claude runs inside a Docker container with ML tools, writes its own Python scripts, and iterates on solutions across sessions.

## Architecture

```
autocomp/
├── spec.md             # This file
├── program.md          # Agent instructions (short)
├── run_session.py      # Agent SDK loop + tools
├── orchestrator.py     # docker build + docker run
├── Dockerfile          # Fat container with ML libs
├── pyproject.toml
└── competitions/
    └── <comp-name>/
        └── journal.md  # Strategy notes + what to try next
```

## Components

### Dockerfile

Fat container with everything the agent needs:

- Python 3.11+
- ML/CV libs: pytorch, scikit-learn, xgboost, lightgbm, catboost, opencv, albumentations, timm, transformers, etc.
- Data tools: pandas, polars, numpy
- Kaggle CLI (for downloading data and submitting)
- wandb (experiment tracking)
- Standard unix tools: curl, wget, git

No custom scripts or helpers. Claude writes all Python itself.

### run_session.py

Agent SDK loop. Gives Claude three tools:

- `bash` — run shell commands
- `write_file` — create/edit files
- `read_file` — read files

System prompt loads `program.md` and the competition's `journal.md` as context.

### orchestrator.py

Minimal. Two operations:

1. `docker build` the image
2. `docker run` a session for a given competition

The human decides when to run sessions. No loop mode, no status command, no retry logic.

### program.md

Short agent instructions covering:

1. Read your journal to understand where you left off
2. Check the leaderboard / current best score
3. Pick an experiment to run
4. Write Python, train, evaluate
5. Track experiments with wandb
6. Submit to Kaggle if results look good
7. Update journal with findings and next steps

### journal.md

Per-competition scratchpad. NOT for detailed metrics (wandb handles that). Contains:

- Competition goal and evaluation metric
- High-level strategy notes
- What the agent has learned so far
- What to try next session

### External knowledge

The agent has curl/wget in the container. It can:

- Read papers/docs via Jina reader (`r.jina.ai/<url>`)
- Search arxiv API
- Read Kaggle discussion pages

No MCP servers or special tool wrappers needed.

### Experiment tracking

wandb handles all experiment tracking:

- Metrics, loss curves, scores
- Hyperparameters
- Run comparisons
- Artifacts (models, predictions)

The agent calls `wandb.init()`, `wandb.log()`, etc. in its own training scripts. We just pre-install it.

## What we explicitly don't build

- **compete.py** — helper functions the agent won't use
- **config.yaml** — second source of truth alongside journal
- **CLAUDE.md** — replaced by system prompt in run_session.py
- **templates/ directory** — journal template inlined in run_session.py
- **Loop/retry logic in orchestrator** — human decides when to run
- **Custom tool wrappers** — agent writes Python directly
