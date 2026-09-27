# ORB Lab: how to open the dashboard

No credit card, no Databento account and no API keys are needed for anything on this page. The only
data the dashboard downloads is free Yahoo Finance data for SPY and QQQ.

> Everything in this dashboard about SPY/QQQ is a **FREE PROXY EXPERIMENT — NOT FUTURES VALIDATION**.

---

## Option A: in your web browser, nothing to install (recommended)

This uses **GitHub Codespaces**, a computer in the cloud that GitHub lends you. Personal GitHub
accounts get a free monthly allowance (about 60 hours on the standard machine) and **do not need a
payment method**. Without one, GitHub simply stops the Codespace when the allowance is used up; it
never charges you.

1. Sign in to GitHub and open the repository page: **https://github.com/s9d9m/backtest**
2. Click the green **`<> Code`** button → the **Codespaces** tab → **Create codespace on
   claude/adoring-keller-3zp5av** (the only branch, already selected).
3. Wait. The first time takes about **3–6 minutes**: a code editor appears in your browser and
   installs everything by itself.
4. The dashboard **opens in a new browser tab by itself** when it is ready.
   - If it doesn't, click the **PORTS** tab at the bottom of the editor and click the little globe
     icon next to **ORB Lab dashboard (8501)**.
   - Your browser may block the pop-up; allow it, or use the PORTS tab.
5. When you are finished, close the tab. To save your free hours, go to
   **https://github.com/codespaces**, click the **⋯** next to your codespace and choose **Stop
   codespace**. Next time, open the same codespace again from that page (it starts much faster).

## Option B: on your own computer

1. Install **Python 3.12** from https://www.python.org/downloads/.
   - Windows: tick **"Add python.exe to PATH"** on the first installer screen.
2. On the GitHub page, click **`<> Code` → Download ZIP**, then unzip it (for example to your
   Desktop).
3. Open the unzipped folder and double-click:
   - **Windows**: `start_dashboard.bat`
   - **Mac**: `start_dashboard.command`
     - If macOS says it can't be opened: right-click it → **Open** → **Open**.
   - **Linux**: run `./start_dashboard.sh` in a terminal.
4. The first time, it installs everything (a few minutes). Then your browser opens
   **http://localhost:8501** with the dashboard.
5. Keep the black window open while you use the dashboard. Close it to stop.

---

## Using the dashboard

The menu on the left is ordered the way research normally flows. You rarely need the last two entries.

| Page | What it is for |
|---|---|
| **Overview** | The home screen: the strategy being tested, the market, your account and risk, the key results, and how far validation has got (in-sample, validation, blind holdout, walk-forward, robustness, execution, Monte Carlo) with a research verdict. |
| **Strategy** | Build or edit the strategy: opening range, entry, stop, target, trade management, direction, **position sizing** (e.g. $40,000 and 1% or 2% risk per trade) and **execution costs** (every $ amount is per share or per contract, per side). Rare settings sit under *Advanced settings*. Click **Save strategy**; every page then uses it. |
| **Backtest** | Did it make or lose money? Balance, return, P&L, trades, win rate, expectancy, profit factor and drawdown first, then equity, drawdown, cumulative R and trade-by-trade charts, then details. |
| **Optimize** | Try many settings at once. It separates the **best historical configuration** (usually luck) from a **robust candidate** (neighbouring settings also work) and shows how much of the result luck alone could explain. Buttons: *Use this strategy*, *Backtest*, *Validate*, *Stress test*. |
| **Validate** | The only place evidence comes from. **Blind holdout test**: hide the last part of the data, check the validation period, freeze the strategy, test it once. **Walk-forward**: choose → freeze → test on the next unseen period → roll forward, in the background; the stitched blind out-of-sample result is what counts. |
| **Stress test** | Parameter robustness (plateau or spike), execution sensitivity (more slippage, higher costs, worse fills) and Monte Carlo (how bad the path could have been). |
| **Results** | One page with every stage and the verdict: PROMISING, MIXED, NO PRELIMINARY EVIDENCE or INSUFFICIENT EVIDENCE. Download it as a report. |
| **Trade explorer** | Every trade, with filters (dates, long/short, winners/losers, entry type, R) and a chart of the chosen trade. CSV download. |
| **Data** | Load or change data: free Yahoo SPY/QQQ, your own CSV/Parquet file, or synthetic test data. Data-quality checks, sessions, provenance, and the (inactive) plan for real futures data. |
| **Settings & research** | Market cost tables, frozen strategies and fingerprints, the holdout ledger, the research registry, software self-checks and the Phase-0 archive. |

### A first session (about 10 minutes)
1. **Overview**: if there is no data yet, click **Load free SPY data** (about 30 seconds, no account).
2. **Validate → Blind holdout test → Save split and withhold the holdout.** Do this before optimising.
3. **Optimize → RUN OPTIMIZATION**, then *Use this strategy* on the robust candidate.
4. **Stress test**: run the three tabs.
5. **Validate**: *Run validation check* → *FREEZE* → tick the box → *RUN BLIND HOLDOUT TEST* (once).
6. **Results**: read the verdict. With ~60 days of free data it will almost always say the evidence is insufficient —
   that is the honest answer, not a fault.

### Good to know
- Hover the small **?** icons for plain-English explanations of every statistic.
- Nothing here can spend money. The paid futures data source (Databento) is prepared but switched off (see **Data**).
- Everything about SPY/QQQ is a **FREE PROXY — NOT FUTURES VALIDATION**.
- Walk-forward runs keep going in the background even if you close the browser tab.
