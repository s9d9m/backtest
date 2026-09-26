# Real market data: source evaluation and acquisition

Status (2026-09-26): **no real market data has been loaded yet.** Nothing in this repository says
anything about ORB profitability. The synthetic tests only validate the engine.

## Requirements

- Genuine CME/COMEX futures prices, not ETF, CFD or spot-FX proxies.
- 1-minute OHLCV. Target 10 years; 5 years is the minimum acceptable.
- Instruments: 6E, GC, ES, NQ.
- Contract identity must be known for every bar, so that no session mixes two contracts and roll
  gaps can never create a breakout signal.
- Timestamps whose timezone is unambiguous across the whole history.

## Candidates

| Source | History (1-min) | Construction | Timezone / stamp | Volume | Cost (indicative) | Access | Verdict |
|---|---|---|---|---|---|---|---|
| **Databento** GLBX.MDP3 | CME Globex from **2010-06-06**. Before 2017-05-21 it is a CME DataMine backfill of the legacy FIX/FAST feed. | **Individual contracts** (raw prices). Continuous symbology `ROOT.v.0` (volume lead, ranked on the *previous day's* volume) / `.c.0` / `.n.0`, **not back-adjusted**. | UTC nanoseconds; `ts_event` = **bar open**. | Exchange-reported trade volume. | Usage-based historical pricing; **$125 free credit** for new accounts. Exact cost for this request not verified: use `cli fetch --estimate-only`. | API key; Python client `databento`. | **Recommended** |
| FirstRate Data | Futures 1-min from ~2007/2008 (vendor claim). | Continuous: unadjusted, absolute-adjusted, ratio-adjusted. Individual contracts in some bundles. The roll rule is vendor-defined. | **Naive US Eastern** timestamps. Zero-volume minutes omitted. | Yes | One-time bundle purchase (price not verified). | Download after purchase. | Acceptable second choice. Needs the individual-contract files (or roll dates), and naive-ET handling must be verified around DST. |
| Kibot | Since 2009 (vendor claim). | Continuous (not back-adjusted) plus individual contracts. | Vendor documentation (ET); **stamp convention must be verified**. | Yes | ~$520 one-time for all futures, 1-min (vendor page). | Download after purchase. | Acceptable second choice, same caveats. |
| Portara / CQG | Deep, institutional quality. | Individual plus several continuous methods. | Documented. | Yes | Expensive. | Purchase. | Good quality; overkill unless cost is irrelevant. |
| IQFeed / broker feeds (IB, NinjaTrader, TradeStation) | IB serves expired futures for only ~2 years; others vary. | Mostly continuous, platform-specific rolls. | Varies. | Yes | Subscription. | Account needed. | **Insufficient** for 10 years of reproducible research. |
| Free (Yahoo, Kaggle/GitHub dumps, HistData, Dukascopy) | Yahoo 1-min covers only days; dumps have unknown provenance; HistData/Dukascopy are **spot FX/CFD** with no exchange volume. | Unknown or not futures. | Unknown or mixed. | Missing or synthetic. | Free | - | **Rejected.** No free source meets the requirements. |

Vendor facts come from vendor pages found via web search on 2026-09-26. This environment's network
policy blocks the vendor domains, so the pages could not be fetched and every claim should be
re-checked on the vendor's site before purchase. Sources:
[Databento CME history extended to 2010](https://databento.com/blog/CME-history-extended-to-2010),
[GLBX.MDP3 dataset](https://databento.com/datasets/GLBX.MDP3),
[continuous symbology](https://databento.com/docs/examples/symbology/continuous),
[continuous contracts guide](https://databento.com/microstructure/continuous-contract),
[OHLCV schema](https://databento.com/docs/schemas-and-data-formats/ohlcv),
[usage pricing & credits](https://databento.com/docs/faqs/usage-pricing-and-data-credits),
[FirstRate GC](https://firstratedata.com/i/futures/GC),
[FirstRate FAQ](https://firstratedata.com/about/FAQ),
[Kibot continuous futures 1-min](https://www.kibot.com/historical-data/continuous-futures-contracts-1-minute-intraday-data.html).

## Recommendation: Databento GLBX.MDP3, `ohlcv-1m`, individual contracts

Why:
1. It is exchange-sourced CME data covering all four instruments, with 15+ years of history, so
   10 years plus a 12-month lockbox is achievable.
2. It gives **raw individual-contract bars**. We build the front-month series ourselves with a
   documented, *causal* roll rule. Every session uses exactly one contract at its real traded
   prices. There is no back-adjustment, so there are no artificial price levels, and no roll gap can
   fall inside a session.
3. UTC timestamps with a documented bar-open convention remove the most common timezone and
   bar-labelling errors.
4. Pay-as-you-go pricing and an exact pre-download cost estimate.

Known caveats (checked by the DQ pipeline, not assumed away):
- **Pre-2017 data is a legacy-feed backfill.** The DQ report is broken down by year so any change in
  character at 2017-05-21 is visible.
- **No-trade minutes produce no bar.** Missing minutes are therefore normal in thin periods. The
  RTH-completeness check separates "no trades" from genuine outages (outages appear as long gaps in
  otherwise active sessions).
- **Parent-symbol requests include calendar spreads.** Spread instruments (for example `ESH4-ESM4`)
  are removed by an explicit outright-only symbol filter before anything else happens.
- **Licensing.** Historical data is licensed for the subscriber's internal use. Do not commit or
  redistribute raw data; `data/` is git-ignored.

### How the front-month series is built (`orb_lab/data_sources/futures_roll.py`)

1. Keep outright contracts only, matching `^ROOT[FGHJKMNQUVXZ]\d{1,2}$`.
2. Assign every bar to its Globex session (18:00 ET → 17:00 ET).
3. For session *d*, trade the outright with the **highest total volume in session d−1**. This uses
   only information available before session *d* opens.
4. Rolls only move forward in expiry: never back to an earlier contract.
5. The first session in the file has no prior session and is dropped.
6. Output: one contract per session, with a `contract` column. Every roll is written to a roll
   report.
7. Daily true range and ATR never use a previous close from a different contract. On roll sessions,
   TR = high − low.

Result: individual-contract prices stitched at session boundaries only. It is **not** back-adjusted
and not ratio-adjusted, which is correct for an intraday strategy.

## What you need to do

1. Create a Databento account at <https://databento.com> ($125 free credit) and copy an API key.
2. Let this cloud environment reach Databento: in the environment settings (cloud environment menu
   in the session title bar → Edit → Network access), allow `hist.databento.com`, or choose a broader
   access level. Add the key as an environment secret named `DATABENTO_API_KEY`.
   *Or* run the fetch command on your own machine and place the resulting Parquet files in `data/`.
3. Check the cost first. Nothing is downloaded or billed:

   ```bash
   python -m orb_lab.cli fetch --instrument 6E --start 2015-06-01 --end 2026-09-01 --estimate-only
   ```

4. Download. This is resumable, fetches year by year, and caches raw per-contract files under
   `data/raw/databento/`:

   ```bash
   for s in 6E GC ES NQ; do
     python -m orb_lab.cli fetch --instrument $s --start 2015-06-01 --end 2026-09-01
   done
   ```

   Each run writes `data/<SYM>_databento_front_1m.parquet`, a roll report and a provenance JSON
   (request parameters, row counts, SHA-256 hashes).
5. Run the data-quality stage (Task 2). It also seals the 12-month lockbox:

   ```bash
   python -m orb_lab.cli dq-report --data data/6E_databento_front_1m.parquet --instrument 6E
   ```

If you prefer FirstRate or Kibot, buy the **individual-contract** files. The loader handles naive
Eastern timestamps (`--tz America/New_York`) and bar-start/bar-end stamps (`--convention`), and
`futures_roll.build_front_month` works on any per-contract frame.
