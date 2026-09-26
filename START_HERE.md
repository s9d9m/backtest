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

The tabs are along the top; the controls are on the left.

1. **PHASE-0 RESULTS** (opens first) shows the finished free experiment without downloading anything:
   - the conclusion for SPY and QQQ;
   - the trade log (downloadable as CSV);
   - charts of individual trades (orange band = opening range, triangle = entry, dashed lines =
     stop/target, X = exit);
   - heatmaps and the cost table.
2. To try your **own** settings on free data:
   1. Leave **Data source = Free Yahoo (SPY/QQQ)** and pick **SPY** or **QQQ**.
   2. Click **Load data**. It uses the saved copy if there is one, otherwise it downloads the last
      ~60 days from Yahoo (about 30 seconds).
   3. Tick "Download fresh data" only if you want the newest days.
   4. **DATA** tab: the data-quality check (it should say 0 errors).
   5. **BACKTEST** tab:
      - choose the opening-range length, entry type, stop, target (R), last entry time and direction;
      - **Commission/side** is the cost per share per fill (default $0.02), and **Shares** is the
        position size;
      - click **RUN BACKTEST** to see the equity curve, statistics and trades. Scroll down for charts
        and the cost-sensitivity button.
   6. **TRADE LOG** tab: every trade of your last backtest (downloadable).
   7. **OPTIMIZATION** tab: tries many settings at once and shows heatmaps.
      - Remember: with only ~40 days of data, the "best" row is mostly luck. Look for broad blue areas
        in the heatmaps, not the single best number.
3. **WALK-FORWARD**, **ROBUSTNESS** and **MONTE CARLO** are for the full futures study later. With
   free data they are empty or informational. That is expected.

### Good to know

- Nothing here can spend money. The only paid data source (Databento) needs a key you have not
  entered, and even then the downloader refuses anything above a budget cap.
- Yahoo only provides about **60 days** of 5-minute bars, so results on this data are too small a
  sample to prove or disprove an edge.
- If a button seems to do nothing, look at the top right of the page. "Running…" means it is still
  working (a big optimization can take a few minutes).
