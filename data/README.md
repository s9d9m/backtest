Place historical intraday OHLCV files (CSV or Parquet) here. Files in this folder are git-ignored.

Required columns (case-insensitive, common aliases accepted): timestamp, open, high, low, close, volume.
Optional: bid, ask, contract, symbol.

Timestamps must either carry a timezone/offset (e.g. `2024-03-11T13:30:00Z`) or you must declare the
source timezone when loading. Naive timestamps are never guessed.

Generate a synthetic file for experimentation:

    python -m orb_lab.cli synth --instrument ES --start 2019-01-01 --end 2023-12-31 --out data/synth_ES.parquet
