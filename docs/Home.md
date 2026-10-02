# Funded account model

Start here. Tasks live in GitHub issues (see the pinned **Roadmap (tracking)** issue). This vault holds the reasoning: methods, rules, decisions, and experiment results.

## Map
- Methods: [[MDP setup]] → [[Post-lock model]] → [[Pre-lock model]] → [[Time layer]] → [[Economics]]
- Checks: [[Baselines]], [[Validation]], [[Out-of-sample test]]
- Firm rules: [[Lucid 25K Flex]]
- Decisions: folder `Decisions/`
- Experiments: folder `Experiments/` (new note: Templates → Experiment)

## Recent experiments
```dataview
TABLE date, change, result
FROM "Experiments"
SORT date DESC
LIMIT 10
```

## Open questions
```dataview
LIST
FROM "Experiments" OR "Rules"
WHERE contains(status, "open")
```
