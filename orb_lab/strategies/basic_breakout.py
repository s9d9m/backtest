"""Basic Opening Range Breakout simulation kernel (numba).

Event order inside every base bar ``i`` (start time ``t``) of a session:

0. If ``t >= exit_min`` (time exit / session end): an open position exits at ``open[i]`` minus slippage.
1. Entry-timeframe buckets that **ended at or before t** are evaluated (close confirmation). A bucket
   covers ``[range_end + k*entry_tf, range_end + (k+1)*entry_tf)`` and its close is only known at its
   end time, so a signal can at the earliest be executed at the open of bar ``i``.
2. A pending limit order is checked against bar ``i``.
3. A resting stop-entry order is checked against bar ``i``.
4. An open position is checked for stop / target / break-even on bar ``i`` (with the entry-bar rules
   described in :mod:`orb_lab.engine.execution`).
5. Bar ``i`` is appended to the current entry-timeframe bucket.

Because a bucket is only evaluated once a *later* bar exists, and fills happen on that later bar, no
decision can use information from the future (see tests/test_lookahead.py).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from ..engine.execution import (
    AMB_CONSERVATIVE,
    AMB_OPTIMISTIC,
    DAY_AMBIGUOUS_ENTRY,
    DAY_BAD_OR,
    DAY_FILTERED,
    DAY_INVALID_STOP,
    DAY_NO_ATR,
    DAY_NO_SIGNAL,
    DAY_NO_TIME,
    DAY_NOT_FILLED,
    DAY_TRADED,
    DIR_LONG,
    DIR_SHORT,
    EM_LIMIT,
    EM_MARKET,
    EM_STOP,
    EPS,
    EXIT_BREAKEVEN,
    EXIT_DATA_END,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIME,
    FILL_CONSERVATIVE,
    SM_ATR,
    compute_stop_target,
    reentry_allowed,
)

N_OUT_FIELDS = 19


@njit(cache=True)
def simulate_basic_breakout(
    tod,
    o,
    h,
    l,
    c,
    day_start,
    day_end,
    d_lo,
    d_hi,
    or_hi,
    or_lo,
    or_ok,
    atr,
    cutoff_min,
    exit_min,
    range_end,
    entry_tf,
    entry_method,
    confirm_ticks,
    confirm_frac,
    buffer_ticks,
    stop_method,
    stop_param,
    target_r,
    direction,
    max_trades,
    reentry,
    be_r,
    or_atr_min,
    or_atr_max,
    slip,
    fill_model,
    ambiguity,
    max_fill_delay,
    per_day_cap,
):
    n_days_sel = d_hi - d_lo
    day_cap = max_trades if max_trades > 0 else per_day_cap
    cap = max(n_days_sel * day_cap, 1)
    t_day = np.empty(cap, np.int64)
    t_dir = np.empty(cap, np.int64)
    t_sig_tod = np.empty(cap, np.int64)
    t_entry_idx = np.empty(cap, np.int64)
    t_exit_idx = np.empty(cap, np.int64)
    t_entry_px = np.empty(cap, np.float64)
    t_entry_slip = np.empty(cap, np.float64)
    t_stop = np.empty(cap, np.float64)
    t_target = np.empty(cap, np.float64)
    t_exit_px = np.empty(cap, np.float64)
    t_exit_slip = np.empty(cap, np.float64)
    t_reason = np.empty(cap, np.int64)
    t_amb = np.empty(cap, np.bool_)
    t_intra = np.empty(cap, np.bool_)
    t_be = np.empty(cap, np.bool_)
    t_mfe = np.empty(cap, np.float64)
    t_mae = np.empty(cap, np.float64)
    t_risk = np.empty(cap, np.float64)
    status = np.full(day_start.shape[0], -1, np.int64)
    n = 0

    need_atr = stop_method == SM_ATR or or_atr_min > 0.0 or or_atr_max > 0.0
    cons_fill = fill_model == FILL_CONSERVATIVE
    through = 1.0 if cons_fill else 0.0
    allow_long = direction != DIR_SHORT
    allow_short = direction != DIR_LONG

    for d in range(d_lo, d_hi):
        ds = day_start[d]
        de = day_end[d]
        if not or_ok[d]:
            status[d] = DAY_BAD_OR
            continue
        H = or_hi[d]
        L = or_lo[d]
        W = H - L
        if W < 1.0 - EPS:
            status[d] = DAY_BAD_OR
            continue
        a = atr[d]
        if need_atr and not (a > 0.0):
            status[d] = DAY_NO_ATR
            continue
        if or_atr_min > 0.0 and W / a < or_atr_min:
            status[d] = DAY_FILTERED
            continue
        if or_atr_max > 0.0 and W / a >= or_atr_max:
            status[d] = DAY_FILTERED
            continue
        xm = exit_min[d]
        cut = cutoff_min[d]
        if cut > xm:
            cut = xm
        if cut <= range_end:
            status[d] = DAY_NO_TIME
            continue

        off = confirm_ticks + confirm_frac * W
        up_lvl = H + off
        dn_lvl = L - off
        trig_up = np.floor(up_lvl + EPS) + 1.0 + buffer_ticks
        trig_dn = np.ceil(dn_lvl - EPS) - 1.0 - buffer_ticks
        lim_up = H + buffer_ticks
        lim_dn = L - buffer_ticks

        i = ds
        while i < de and tod[i] < range_end:
            i += 1

        day_status = DAY_NO_SIGNAL
        pos = 0
        n_today = 0
        last_dir = 0
        last_loss = False
        last_exit_tod = -1000000
        last_exit_idx = -1
        pend = 0
        pend_lvl = 0.0
        pend_sig_tod = 0
        b_end = range_end + entry_tf
        b_has = False
        b_close = 0.0
        armed_up = True
        armed_dn = True
        e_px = 0.0
        e_slip = 0.0
        stp = 0.0
        tgt = 0.0
        has_tgt = False
        risk = 0.0
        e_idx = -1
        e_intra = False
        be_done = False
        mfe = 0.0
        mae = 0.0
        sig_tod = 0
        ambiguous_day = False

        while i < de:
            t = tod[i]
            # ---------------------------------------------------- 0. time exit
            if t >= xm:
                if pos != 0:
                    xpx = o[i] - pos * slip
                    realized = (xpx - e_px) * pos
                    if realized > mfe:
                        mfe = realized
                    if -realized > mae:
                        mae = -realized
                    t_day[n] = d
                    t_dir[n] = pos
                    t_sig_tod[n] = sig_tod
                    t_entry_idx[n] = e_idx
                    t_exit_idx[n] = i
                    t_entry_px[n] = e_px
                    t_entry_slip[n] = e_slip
                    t_stop[n] = stp
                    t_target[n] = tgt if has_tgt else np.nan
                    t_exit_px[n] = xpx
                    t_exit_slip[n] = slip
                    t_reason[n] = EXIT_TIME
                    t_amb[n] = False
                    t_intra[n] = e_intra
                    t_be[n] = be_done
                    t_mfe[n] = mfe
                    t_mae[n] = mae
                    t_risk[n] = risk
                    n += 1
                    pos = 0
                break

            can_trade = n_today < day_cap
            # ------------------------------------- 1. completed close-confirmation buckets
            if entry_method != EM_STOP:
                while t >= b_end:
                    if b_has and pos == 0 and can_trade and b_end <= cut and b_end > last_exit_tod:
                        sig = 0
                        if b_close > up_lvl + EPS:
                            if allow_long and reentry_allowed(1, n_today, last_dir, last_loss, reentry):
                                sig = 1
                        elif b_close < dn_lvl - EPS:
                            if allow_short and reentry_allowed(-1, n_today, last_dir, last_loss, reentry):
                                sig = -1
                        if sig != 0:
                            if entry_method == EM_MARKET:
                                if t - b_end <= max_fill_delay:
                                    fill = o[i] + sig * slip
                                    s_, t_, ht_, r_, ok_ = compute_stop_target(
                                        sig, fill, H, L, W, a, stop_method, stop_param, target_r
                                    )
                                    if ok_:
                                        pos = sig
                                        e_px = fill
                                        e_slip = slip
                                        stp = s_
                                        tgt = t_
                                        has_tgt = ht_
                                        risk = r_
                                        e_idx = i
                                        e_intra = False
                                        be_done = False
                                        mfe = 0.0
                                        mae = 0.0
                                        sig_tod = b_end
                                        day_status = DAY_TRADED
                                    elif day_status != DAY_TRADED:
                                        day_status = DAY_INVALID_STOP
                                elif day_status != DAY_TRADED:
                                    day_status = DAY_NOT_FILLED
                            else:
                                if pend != sig:
                                    pend = sig
                                    pend_lvl = lim_up if sig > 0 else lim_dn
                                    pend_sig_tod = b_end
                                    if day_status != DAY_TRADED:
                                        day_status = DAY_NOT_FILLED
                    b_has = False
                    b_end += entry_tf

            # ---------------------------------------------------- 2. pending limit order
            if pos == 0 and pend != 0:
                if t >= cut or not can_trade:
                    pend = 0
                else:
                    filled = False
                    fill = 0.0
                    intra = True
                    if pend > 0:
                        if l[i] <= pend_lvl - through + EPS:
                            filled = True
                            if cons_fill or o[i] > pend_lvl:
                                fill = pend_lvl
                            else:
                                fill = o[i]
                                intra = False
                    else:
                        if h[i] >= pend_lvl + through - EPS:
                            filled = True
                            if cons_fill or o[i] < pend_lvl:
                                fill = pend_lvl
                            else:
                                fill = o[i]
                                intra = False
                    if filled:
                        sig = pend
                        pend = 0
                        s_, t_, ht_, r_, ok_ = compute_stop_target(sig, fill, H, L, W, a, stop_method, stop_param, target_r)
                        if ok_:
                            pos = sig
                            e_px = fill
                            e_slip = 0.0
                            stp = s_
                            tgt = t_
                            has_tgt = ht_
                            risk = r_
                            e_idx = i
                            e_intra = intra
                            be_done = False
                            mfe = 0.0
                            mae = 0.0
                            sig_tod = pend_sig_tod
                            day_status = DAY_TRADED
                        elif day_status != DAY_TRADED:
                            day_status = DAY_INVALID_STOP

            # ---------------------------------------------------- 3. stop entry (intrabar)
            if entry_method == EM_STOP and pos == 0 and t < cut and i > last_exit_idx and can_trade:
                go_l = allow_long and armed_up and h[i] >= trig_up - EPS
                go_s = allow_short and armed_dn and l[i] <= trig_dn + EPS
                if go_l:
                    go_l = reentry_allowed(1, n_today, last_dir, last_loss, reentry)
                if go_s:
                    go_s = reentry_allowed(-1, n_today, last_dir, last_loss, reentry)
                if go_l and go_s:
                    if o[i] >= trig_up - EPS:
                        go_s = False
                    elif o[i] <= trig_dn + EPS:
                        go_l = False
                    else:
                        # both sides triggered inside one bar: order unknowable -> stand aside for the day
                        ambiguous_day = True
                        if day_status != DAY_TRADED:
                            day_status = DAY_AMBIGUOUS_ENTRY
                        break
                if go_l or go_s:
                    sig = 1 if go_l else -1
                    if sig > 0:
                        base_px = o[i] if o[i] > trig_up else trig_up
                        intra = o[i] < trig_up - EPS
                    else:
                        base_px = o[i] if o[i] < trig_dn else trig_dn
                        intra = o[i] > trig_dn + EPS
                    fill = base_px + sig * slip
                    s_, t_, ht_, r_, ok_ = compute_stop_target(sig, fill, H, L, W, a, stop_method, stop_param, target_r)
                    if ok_:
                        pos = sig
                        e_px = fill
                        e_slip = slip
                        stp = s_
                        tgt = t_
                        has_tgt = ht_
                        risk = r_
                        e_idx = i
                        e_intra = intra
                        be_done = False
                        mfe = 0.0
                        mae = 0.0
                        sig_tod = t
                        day_status = DAY_TRADED
                    else:
                        if day_status != DAY_TRADED:
                            day_status = DAY_INVALID_STOP
                        if sig > 0:
                            armed_up = False
                        else:
                            armed_dn = False

            # ---------------------------------------------------- 4. manage open position
            if pos != 0:
                s = pos
                if s > 0:
                    fo = o[i]
                    fh = h[i]
                    fl = l[i]
                    fc = c[i]
                else:
                    fo = -o[i]
                    fh = -l[i]
                    fl = -h[i]
                    fc = -c[i]
                E = s * e_px
                S = s * stp
                T = s * tgt
                entry_intra_bar = i == e_idx and e_intra
                reason = 0
                xpx_n = 0.0
                xslip = 0.0
                amb = False
                if not entry_intra_bar:
                    if fo <= S + EPS:
                        reason = EXIT_STOP
                        xpx_n = fo - slip
                        xslip = slip
                    elif has_tgt and fo >= T + through - EPS:
                        reason = EXIT_TARGET
                        xpx_n = fo if ambiguity == AMB_OPTIMISTIC else T
                if reason == 0:
                    s_hit = fl <= S + EPS
                    t_hit = has_tgt and fh >= T + through - EPS
                    if entry_intra_bar:
                        if ambiguity == AMB_OPTIMISTIC:
                            s_hit = fc <= S + EPS
                        elif ambiguity == AMB_CONSERVATIVE:
                            t_hit = False
                    if s_hit and t_hit:
                        amb = True
                        if ambiguity == AMB_OPTIMISTIC:
                            reason = EXIT_TARGET
                            xpx_n = T
                        else:
                            reason = EXIT_STOP
                            xpx_n = S - slip
                            xslip = slip
                    elif s_hit:
                        reason = EXIT_STOP
                        xpx_n = S - slip
                        xslip = slip
                    elif t_hit:
                        reason = EXIT_TARGET
                        xpx_n = T
                if reason != 0:
                    if reason == EXIT_STOP and be_done:
                        reason = EXIT_BREAKEVEN
                    realized = xpx_n - E
                    if realized > mfe:
                        mfe = realized
                    if -realized > mae:
                        mae = -realized
                    t_day[n] = d
                    t_dir[n] = s
                    t_sig_tod[n] = sig_tod
                    t_entry_idx[n] = e_idx
                    t_exit_idx[n] = i
                    t_entry_px[n] = e_px
                    t_entry_slip[n] = e_slip
                    t_stop[n] = stp
                    t_target[n] = tgt if has_tgt else np.nan
                    t_exit_px[n] = s * xpx_n
                    t_exit_slip[n] = xslip
                    t_reason[n] = reason
                    t_amb[n] = amb
                    t_intra[n] = e_intra
                    t_be[n] = be_done
                    t_mfe[n] = mfe
                    t_mae[n] = mae
                    t_risk[n] = risk
                    n += 1
                    pos = 0
                    n_today += 1
                    last_dir = s
                    last_loss = realized <= EPS
                    last_exit_tod = t
                    last_exit_idx = i
                    armed_up = False
                    armed_dn = False
                else:
                    if entry_intra_bar:
                        fav = fc - E
                        adv = E - fc
                    else:
                        fav = fh - E
                        adv = E - fl
                    if fav > mfe:
                        mfe = fav
                    if adv > mae:
                        mae = adv
                    if be_r > 0.0 and not be_done and not entry_intra_bar and fh >= E + be_r * risk - EPS:
                        if E > S:
                            stp = s * E
                        be_done = True

            # ---------------------------------------------------- 5. bookkeeping
            if entry_method == EM_STOP and pos == 0:
                if c[i] < trig_up - EPS:
                    armed_up = True
                if c[i] > trig_dn + EPS:
                    armed_dn = True
            b_has = True
            b_close = c[i]
            if pos == 0 and pend == 0 and (t >= cut or n_today >= day_cap):
                break
            i += 1

        if pos != 0:
            last = i - 1 if i >= de else i
            if last < ds:
                last = ds
            xpx = c[last] - pos * slip
            realized = (xpx - e_px) * pos
            if realized > mfe:
                mfe = realized
            if -realized > mae:
                mae = -realized
            t_day[n] = d
            t_dir[n] = pos
            t_sig_tod[n] = sig_tod
            t_entry_idx[n] = e_idx
            t_exit_idx[n] = last
            t_entry_px[n] = e_px
            t_entry_slip[n] = e_slip
            t_stop[n] = stp
            t_target[n] = tgt if has_tgt else np.nan
            t_exit_px[n] = xpx
            t_exit_slip[n] = slip
            t_reason[n] = EXIT_DATA_END
            t_amb[n] = False
            t_intra[n] = e_intra
            t_be[n] = be_done
            t_mfe[n] = mfe
            t_mae[n] = mae
            t_risk[n] = risk
            n += 1
        if ambiguous_day and day_status == DAY_TRADED:
            day_status = DAY_TRADED
        status[d] = day_status

    return (
        n,
        status,
        t_day[:n],
        t_dir[:n],
        t_sig_tod[:n],
        t_entry_idx[:n],
        t_exit_idx[:n],
        t_entry_px[:n],
        t_entry_slip[:n],
        t_stop[:n],
        t_target[:n],
        t_exit_px[:n],
        t_exit_slip[:n],
        t_reason[:n],
        t_amb[:n],
        t_intra[:n],
        t_be[:n],
        t_mfe[:n],
        t_mae[:n],
        t_risk[:n],
    )
