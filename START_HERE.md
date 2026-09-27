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

The tabs are along the top; the data controls are on the left. Everything works in the browser; you do not
need the terminal.

### 1. Load data (left sidebar)
- Leave **Data source = Free Yahoo (SPY/QQQ)**, pick **SPY** or **QQQ**, click **Load data**. It uses the
  saved copy if there is one, otherwise downloads the last ~60 days (about 30 seconds). Tick "Download fresh
  data" only if you want the newest days.
- **Synthetic** data is for testing the software only (a yellow banner says so).

### 2. PIPELINE tab: the guided workflow
It shows each stage with ✅ / ⬜ / ⚠️:
**DATA → SPLIT → BACKTEST / OPTIMIZE → VALIDATE → FREEZE → WALK-FORWARD → ROBUSTNESS + STRESS → MONTE CARLO → BLIND HOLDOUT → REPORT**
1. **Save split and withhold the holdout** (do this first). The last 20 % of days becomes a *blind
   holdout* that no other tab can see.
2. Explore in **BACKTEST**, search in **OPTIMIZATION** (it defaults to the *train* days only).
3. Back in PIPELINE: **Evaluate on train and validation**, pick one, **Use as candidate**.
4. **FREEZE** it (creates a locked file with a fingerprint).
5. **RUN BLIND HOLDOUT TEST** once. Only the first test counts as blind.

### 3. The other tabs
- **BACKTEST**: one parameter set.
  - Costs are labelled with units: every $ amount is **per share (or per contract) per side**, e.g.
    $0.02/share friction = $0.04 per share per round trip.
  - Sizing: fixed quantity, or **risk % per trade** (0.25 / 0.5 / 1 / 2 % or custom) of a starting equity
    such as $40,000. The trade log shows shares/contracts and $ at risk for every trade.
  - Under the numbers: **"What do these numbers mean?"** explains every metric and warns when the sample
    is too small. Open **"Equity & risk view"** for equity, drawdown, cumulative R, trade-by-trade P&L and
    position size. **Run execution stress test** re-runs the setup with worse fills and higher costs and
    says ROBUST, FRAGILE or NOT POSITIVE.
- **OPTIMIZATION**: tries many settings. Choose a *reduced* preset; a *comprehensive* search needs an
  extra tick-box. The table shows how many configurations were searched, how many "effective" independent
  tries that is, what the best result would look like **by luck alone**, and whether each row is a
  *plateau* (neighbours also work) or a *spike* (probably luck). **Use as candidate** sends a row to the
  other tabs.
- **WALK-FORWARD**: open "Configure and launch", choose the window unit (use **trading sessions** for the
  free ~60-day data; the **months** presets 12/3/3/3, 24/3/3, 24/6/6, 36/6/6 are for the real futures
  data later), a parameter space and costs, then **LAUNCH**. It runs in the background with a progress
  bar; you can **Stop** and **Resume**. Only the blue **blind OOS** parts are evidence; training and
  validation numbers are not.
- **ROBUSTNESS**: pick the candidate, **RUN NEIGHBOURHOOD SWEEP**. Each chart changes one setting
  (orange bar = your candidate). Verdict PLATEAU is good, SPIKE means fragile. You can also draw a
  two-setting heatmap.
- **MONTE CARLO**: choose which trades to simulate (candidate, last backtest, walk-forward OOS or blind
  holdout), number of simulations and method, then **RUN**. Orange = what happened, grey = simulated.
- **TRADE LOG**: every trade of the last backtest with size, risk and each cost (CSV download).
- **DIAGNOSTICS**: software self-checks (random-walk data must show no edge, planted edge must be found,
  deleting future data must not change past trades, cost units, full test suite). Not market results.
- **REPORT**: one page per candidate across all stages, with a verdict: PROMISING, MIXED, NO PRELIMINARY
  EVIDENCE or INSUFFICIENT EVIDENCE. Download it as Markdown.
- **PHASE-0 RESULTS**: the finished free SPY/QQQ experiment (read-only).

### Good to know
- Nothing here can spend money. The paid futures data source (Databento) is prepared but switched off;
  see "Real futures data" at the bottom of PIPELINE.
- Yahoo only provides about **60 days** of 5-minute bars. Every result on this data is far too small a
  sample to prove or disprove an edge, and the report will say so.
- If a button seems to do nothing, look at the top right of the page. "Running…" means it is still
  working. Walk-forward runs keep going in the background even if you close the tab.
