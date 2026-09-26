# Research record

Committed, append-only evidence trail for the real-market ORB study. Nothing here is edited after it
is written.

- `registry.jsonl`: every hypothesis (with status and the data seen when it was formed), the WFO
  design, every data-quality verdict and every walk-forward experiment (space fingerprint, number of
  configurations, structures, selection rule, run directory). Append-only; timestamps are UTC.
- `lockbox.json`: the per-instrument final lockbox, sealed by the DQ stage. It records when it was
  unlocked, which frozen candidate file (SHA-256) was used, and the single result.
- `data_quality/`: the DQ report for each instrument.

Hypothesis statuses: `PRE_REGISTERED` (stated before seeing the data it is tested on), `EXPLORATORY`
(found after looking at results; it cannot become confirmed without new untouched data),
`CONFIRMED_OOS`, `REJECTED`.

    python -m orb_lab.cli hypothesis list
    python -m orb_lab.cli lockbox-status
