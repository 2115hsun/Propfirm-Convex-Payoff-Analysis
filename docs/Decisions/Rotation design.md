# Rotation: 5 accounts, 1 trade per account per day

**Decision**: each signal goes to one account; each account takes at most one trade a day.
**Why**: about 5 signals a day; one trade a day makes per-trade trailing equal to end-of-day trailing.
**Revisit when**: the day-level model (GitHub issue 5) shows whether several trades a day per account is better.
