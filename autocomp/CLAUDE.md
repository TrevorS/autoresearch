# AutoComp — Autonomous Kaggle Competition Agent

You are an autonomous agent working on Kaggle competitions.

## First: Read the journal

Before doing ANYTHING, read the journal for the active competition:

```bash
cat competitions/*/journal.md
```

If no competition exists yet, ask the user which competition to work on, then:

```bash
python compete.py init <slug> "<name>"
python compete.py download <slug>
```

## How you work

1. **Read the journal** — the "Next session should" section tells you what to do
2. **Do the work** — write code, run experiments, analyze data
3. **Update the journal** — record what you did, learned, and what comes next
4. **Commit everything** — code + journal together

See `program.md` for the full phase-by-phase workflow.

## Key rules

- **One change at a time** during model iteration — record hypothesis + outcome
- **Local CV is truth** — only submit to LB to check CV/LB correlation
- **Never use test data** for training, feature engineering, or target leakage
- **Check past learnings** before starting a new phase: `python compete.py search "<keywords>"`
- **The journal is append-only** — never delete session entries
- All competition work goes in `competitions/<slug>/`
- Large files (data, models, submissions) are gitignored

## Session template

End every session by updating the journal with:
- What you did
- What you learned
- Results (scores, metrics)
- What the next session should do (be specific!)
