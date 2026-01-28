from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Any, Callable, Coroutine, List, Optional

import pandas as pd
import pytz


@dataclass
class BacktestParams:
    ticker: str
    start_date: str  # YYYY-MM-DD (interpreted in US/Eastern trading calendar)
    daily_budget_usd: float
    sell_r: float = 0.10  # also big-buy limit multiple
    fx_krw_per_usd: float = 1450.0
    prefer_avg_buy: bool = True
    intraday_interval: str = "5m"  # e.g. "1m","2m","5m","15m","30m","60m","90m"


@dataclass
class BacktestResult:
    sold: bool
    sell_dt_et: Optional[str]
    sell_price: Optional[float]
    days_to_sell: Optional[int]

    ticker: str
    first_buy_dt_et: str
    first_buy_price: float

    final_shares: int
    final_avg_cost: float
    total_cost_usd: float

    proceeds_usd: Optional[float]
    profit_usd: Optional[float]
    profit_pct_on_cost: Optional[float]

    proceeds_krw: Optional[float]
    profit_krw: Optional[float]

    # For transparency
    logs: List[str]


def _is_valid_interval(interval: str) -> bool:
    valid = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
    return interval in valid


def _parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _fmt_money(x: float) -> str:
    return f"{x:,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%"


from math import floor

@dataclass
class BacktestSeriesResult:
    completed_trades: List[BacktestResult]
    open_position_result: Optional[BacktestResult]
    start_date: date
    end_date: date
    daily_budget_usd: float
    last_day_close_price: Optional[float]


async def _find_next_available_intraday_start(
    start_date: date,
    fetch_intraday: Callable[[str, str, str, str], Coroutine[Any, Any, pd.DataFrame]],
    log: Callable[[str], None],
    probe_ticker: str = "MSFT",
    interval: str = "5m"
) -> date:
    et = pytz.timezone("America/New_York")
    
    days_ago_60 = date.today() - timedelta(days=60)
    
    current_date = start_date
    if start_date < days_ago_60:
        log(f"[DATA] Start date is older than 60 days. Trying to find a recent valid trading day, starting from {days_ago_60}.")
        current_date = days_ago_60
    else:
        log(f"[DATA] Start date is within 60 days. Trying to find a valid trading day, starting from the day after {start_date}.")
        current_date += timedelta(days=1)

    for i in range(365): # Limit to 1 year of searching
        log(f"[DATA] Probing for intraday data on {current_date.isoformat()} with {probe_ticker}...")
        
        intraday_start = current_date.strftime("%Y-%m-%d")
        intraday_end = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
        
        try:
            probe_data = await fetch_intraday(probe_ticker, intraday_start, intraday_end, interval)
            if not probe_data.empty:
                log(f"[DATA] Found available intraday data on {current_date.isoformat()}.")
                return current_date
        except Exception as e:
            log(f"[DATA] Error probing for data on {current_date.isoformat()}: {e}")

        current_date += timedelta(days=1)
        if current_date > date.today() + timedelta(days=1):
             raise RuntimeError("Could not find any recent trading day with intraday data.")

    raise RuntimeError("Could not find a valid trading day with intraday data within a year.")


async def run_backtest(
    params: BacktestParams,
    fetch_daily: Callable[[str, str, str], Coroutine[Any, Any, pd.DataFrame]],
    fetch_intraday: Callable[[str, str, str, str], Coroutine[Any, Any, pd.DataFrame]],
    log: Callable[[str], None],
) -> BacktestSeriesResult:
    et = pytz.timezone("America/New_York")
    logs: List[str] = []

    def _log(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[{ts} UTC] {msg}"
        logs.append(log_msg)
        log(log_msg)

    _log("Backtest run started.")
    try:
        # Input validation
        if params.daily_budget_usd <= 0: raise ValueError("daily_budget_usd must be > 0.")
        if params.sell_r <= 0: raise ValueError("sell_r must be > 0.")
        if not _is_valid_interval(params.intraday_interval): raise ValueError(f"Unsupported interval.")
    except ValueError as e:
        _log(f"[VALIDATION] Input error: {e}")
        raise

    ticker = params.ticker.upper().strip()
    start_d = _parse_yyyy_mm_dd(params.start_date)
    daily_start = (start_d - timedelta(days=7)).strftime("%Y-%m-%d")
    daily_end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    _log(f"[DATA] Downloading daily data for {ticker}...")
    daily = await fetch_daily(ticker, daily_start, daily_end)
    if daily.empty:
        raise RuntimeError("No daily data. Check ticker or date range.")
    
    daily = daily.copy()
    daily["__date"] = pd.to_datetime(daily.index).date
    daily = daily.sort_values("__date")
    
    sim_daily = daily[daily["__date"] >= start_d].copy()
    if sim_daily.empty:
        raise RuntimeError("No trading days found after the start date.")

    intraday_start = sim_daily.iloc[0]["__date"].strftime("%Y-%m-%d")
    intraday_end = (sim_daily.iloc[-1]["__date"] + timedelta(days=1)).strftime("%Y-%m-%d")
    
    _log(f"[DATA] Downloading intraday data ({params.intraday_interval}) for {ticker}...")
    intra = await fetch_intraday(ticker, intraday_start, intraday_end, params.intraday_interval)
    if intra.empty:
        _log(f"[DATA] No intraday data found for {ticker} on {intraday_start}. Searching for the next available day...")
        new_start_date = await _find_next_available_intraday_start(
            start_date=_parse_yyyy_mm_dd(intraday_start),
            fetch_intraday=fetch_intraday,
            log=_log,
            interval=params.intraday_interval
        )
        
        # We need to re-fetch daily data as well to ensure we have the correct data for the new start date
        start_d = new_start_date
        daily_start = (start_d - timedelta(days=7)).strftime("%Y-%m-%d")
        daily_end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

        _log(f"[DATA] Re-downloading daily data for {ticker} from {daily_start}...")
        daily = await fetch_daily(ticker, daily_start, daily_end)
        if daily.empty:
            raise RuntimeError("No daily data after finding a valid intraday start date.")
        
        daily = daily.copy()
        daily["__date"] = pd.to_datetime(daily.index).date
        daily = daily.sort_values("__date")
        
        sim_daily = daily[daily["__date"] >= start_d].copy()
        if sim_daily.empty:
            raise RuntimeError("No trading days found after the adjusted start date.")

        intraday_start = sim_daily.iloc[0]["__date"].strftime("%Y-%m-%d")
        intraday_end = (sim_daily.iloc[-1]["__date"] + timedelta(days=1)).strftime("%Y-%m-%d")
        
        _log(f"[DATA] Re-downloading intraday data ({params.intraday_interval}) for {ticker} from {intraday_start}...")
        intra = await fetch_intraday(ticker, intraday_start, intraday_end, params.intraday_interval)
        if intra.empty:
            raise RuntimeError(f"Still no intraday data for {ticker} even after finding a valid trading day.")

    intra = intra.copy()
    intra.index = pd.to_datetime(intra.index).tz_convert(et)
    intra["__date"] = intra.index.date
    highs_by_date = {d: grp["High"].astype(float) for d, grp in intra.groupby("__date")}

    completed_trades: List[BacktestResult] = []
    shares, total_cost, avg_cost = 0, 0.0, 0.0
    first_buy_dt_et_current, first_trade_date_current, first_open_current = "", None, 0.0

    sim_days = list(sim_daily["__date"].tolist())
    _log(f"[SIM] Simulation started. Looping through {len(sim_days)} trading days...")

    for day_idx, d in enumerate(sim_days):
        if shares == 0:
            day_row = sim_daily[sim_daily["__date"] == d]
            if day_row.empty: continue
            
            open_price = float(day_row.iloc[0]["Open"])
            shares, total_cost, avg_cost = 1, open_price, open_price
            first_trade_date_current = d
            first_open_current = open_price
            first_buy_dt_et_current = f"{d.isoformat()} 09:30 ET (open fill)"
            
            _log(f"[{'INIT' if not completed_trades else 'RE-ENTRY'}] Buy 1 share @ Open {_fmt_money(open_price)} on {d.isoformat()}.")
            continue

        A = avg_cost
        target = A * (1.0 + params.sell_r)
        _log(f"[DAY {day_idx+1}/{len(sim_days)}] {d.isoformat()} | shares={shares}, avg_cost={_fmt_money(A)} | target={_fmt_money(target)}")

        highs = highs_by_date.get(d)
        if highs is not None and not highs.empty and float(highs.max()) >= target:
            sell_price = target
            proceeds = shares * sell_price
            profit = proceeds - total_cost

            trade_result = BacktestResult(
                sold=True, sell_dt_et=f"{d.isoformat()}", sell_price=sell_price,
                days_to_sell=(d - first_trade_date_current).days if first_trade_date_current else None,
                ticker=ticker, first_buy_dt_et=first_buy_dt_et_current, first_buy_price=first_open_current,
                final_shares=shares, final_avg_cost=avg_cost, total_cost_usd=total_cost,
                proceeds_usd=proceeds, profit_usd=profit,
                profit_pct_on_cost=(profit / total_cost) if total_cost else None,
                proceeds_krw=proceeds * params.fx_krw_per_usd, profit_krw=profit * params.fx_krw_per_usd, logs=[]
            )
            completed_trades.append(trade_result)
            _log(f"✅ [SELL] Triggered. Sold {shares} shares at {_fmt_money(sell_price)}.")
            
            shares, total_cost, avg_cost = 0, 0.0, 0.0
            continue

        day_row = sim_daily[sim_daily["__date"] == d]
        if day_row.empty: continue
        C = float(day_row.iloc[0]["Close"])
        
        X, L_avg, L_big = params.daily_budget_usd, A, A * (1.0 + params.sell_r)
        pair_cost = L_avg + L_big
        n_pair = floor(X / pair_cost) if pair_cost > 0 else 0
        rem, ord_avg, ord_big = X - n_pair * pair_cost, n_pair, n_pair

        if params.prefer_avg_buy:
            if rem >= L_avg: ord_avg += 1
        elif rem >= L_big:
            ord_big += 1
        
        fill_avg, fill_big = 0, 0
        if C <= A: fill_avg, fill_big = ord_avg, ord_big
        elif A < C <= L_big: fill_big = ord_big

        buy_shares = fill_avg + fill_big
        if buy_shares > 0:
            buy_cost = buy_shares * C
            _log(f"[BUY] Filled: {buy_shares} shares at Close={_fmt_money(C)}, cost=${_fmt_money(buy_cost)}.")
            shares += buy_shares
            total_cost += buy_cost
            avg_cost = total_cost / shares

    open_pos = None
    if shares > 0:
        open_pos = BacktestResult(
            sold=False, ticker=ticker, first_buy_dt_et=first_buy_dt_et_current,
            first_buy_price=first_open_current, final_shares=shares,
            final_avg_cost=avg_cost, total_cost_usd=total_cost,
            sell_dt_et=None, sell_price=None, days_to_sell=None, proceeds_usd=None,
            profit_usd=None, profit_pct_on_cost=None, proceeds_krw=None, profit_krw=None, logs=[]
        )
    
    last_day_close_price = float(sim_daily.iloc[-1]["Close"]) if not sim_daily.empty else None

    return BacktestSeriesResult(
        completed_trades=completed_trades,
        open_position_result=open_pos,
        start_date=start_d,
        end_date=sim_daily.iloc[-1]["__date"],
        daily_budget_usd=params.daily_budget_usd,
        last_day_close_price=last_day_close_price
    )
