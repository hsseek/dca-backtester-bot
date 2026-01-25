from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

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


def run_backtest(
    params: BacktestParams,
    fetch_daily: Callable[[str, str, str], pd.DataFrame],
    fetch_intraday: Callable[[str, str, str, str], pd.DataFrame],
    log: Callable[[str], None],
) -> BacktestResult:
    """
    Strategy (as finalized by user):
    - Day 0 (start): Buy 1 share at start day's Open. No buy/sell rules applied on start day.
    - From next trading day onward:
      - Compute A = avg_cost at day start.
      - Create LOC buy orders based on A and daily budget X:
        L_avg = A
        L_big = A*(1+r)
        pair_cost = L_avg + L_big
        n_pair = floor(X / pair_cost)
        rem = X - n_pair*pair_cost
        ord_avg = n_pair
        ord_big = n_pair
        Odd-share handling:
          if prefer_avg_buy: if rem >= L_avg then ord_avg += 1 else no extra
          else: if rem >= L_big then ord_big += 1 else no extra (do not fallback to avg)
      - Sell: intraday bars; if High >= target(A*(1+r)) then sell at target immediately and stop.
        Ignore that day's close buys when sell happens.
      - If not sold, then evaluate Close C:
        if C <= A: fill_avg=ord_avg and fill_big=ord_big
        elif A < C <= A*(1+r): fill_avg=0 and fill_big=ord_big
        else: fill none
        All fills are at Close price C.
    """
    et = pytz.timezone("America/New_York")
    logs: List[str] = []

    def _log(msg: str) -> None:
        # Prepend timestamp to all logs for clarity
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_msg = f"[{ts} UTC] {msg}"
        logs.append(log_msg)
        log(log_msg)

    _log("Backtest run started.")
    try:
        if params.daily_budget_usd <= 0:
            raise ValueError("daily_budget_usd must be > 0.")
        if params.sell_r <= 0:
            raise ValueError("sell_r must be > 0.")
        if params.fx_krw_per_usd <= 0:
            raise ValueError("fx_krw_per_usd must be > 0.")
        if not _is_valid_interval(params.intraday_interval):
            raise ValueError(f"Unsupported intraday_interval='{params.intraday_interval}'. Use one of: 1m,2m,5m,15m,30m,60m,90m.")
    except ValueError as e:
        _log(f"[VALIDATION] Input error: {e}")
        raise

    ticker = params.ticker.upper().strip()
    start_d = _parse_yyyy_mm_dd(params.start_date)

    daily_start = (start_d - timedelta(days=7)).strftime("%Y-%m-%d")
    daily_end = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    _log(f"[DATA] Starting daily download for {ticker} from {daily_start} to {daily_end}...")
    daily = fetch_daily(ticker, daily_start, daily_end)
    if daily is None or daily.empty:
        _log("[ERROR] Daily data download failed: no data returned.")
        raise RuntimeError("No daily data returned. Check ticker symbol or data availability.")
    _log(f"[DATA] Daily download done, rows={len(daily)}.")

    daily = daily.copy()
    if "Open" not in daily.columns or "Close" not in daily.columns:
        raise RuntimeError("Daily data missing Open/Close columns.")
    daily = daily.dropna(subset=["Open", "Close"])
    daily["__date"] = pd.to_datetime(daily.index).date
    daily = daily.sort_values("__date")

    daily_after = daily[daily["__date"] >= start_d]
    if daily_after.empty:
        _log(f"[ERROR] No trading days found on or after requested start_date={start_d.isoformat()}.")
        raise RuntimeError("Start date is after the available daily data range.")
    first_row = daily_after.iloc[0]
    first_trade_date: date = first_row["__date"]
    first_open = float(first_row["Open"])

    first_buy_dt_et = f"{first_trade_date.isoformat()} 09:30 ET (assumed open fill)"
    _log(f"[INIT] Start date requested: {start_d.isoformat()}. First trading day found: {first_trade_date.isoformat()}.")
    _log(f"[INIT] Buying 1 share at Open={_fmt_money(first_open)} on {first_trade_date.isoformat()} (rules start next trading day).")

    shares = 1
    total_cost = first_open
    avg_cost = total_cost / shares

    sim_daily = daily[daily["__date"] >= first_trade_date].copy()
    if len(sim_daily) < 2:
        _log("[ERROR] Not enough trading days after start to simulate.")
        raise RuntimeError("Not enough trading days after start to simulate.")

    sim_start_date = sim_daily.iloc[1]["__date"]
    sim_end_date = sim_daily.iloc[-1]["__date"]

    intraday_start = sim_start_date.strftime("%Y-%m-%d")
    intraday_end = (sim_end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    _log(f"[DATA] Starting intraday download ({params.intraday_interval}) for {ticker} from {intraday_start} to {intraday_end}...")
    intra = fetch_intraday(ticker, intraday_start, intraday_end, params.intraday_interval)
    if intra is None or intra.empty:
        _log("[ERROR] Intraday data download failed: no data returned.")
        raise RuntimeError(
            "No intraday data returned. This may be due to Yahoo's lookback limits for the chosen interval. "
            "Try a more recent start date or a coarser interval (e.g., 15m, 30m, 60m)."
        )
    _log(f"[DATA] Intraday download done, rows={len(intra)}.")

    intra = intra.copy()
    if intra.index.tz is None:
        intra.index = pd.to_datetime(intra.index).tz_localize("UTC").tz_convert(et)
    else:
        intra.index = intra.index.tz_convert(et)

    if "High" not in intra.columns:
        raise RuntimeError("Intraday data missing High column.")
    intra = intra.dropna(subset=["High"])
    intra["__date"] = intra.index.date

    highs_by_date: Dict[date, pd.Series] = {d: grp["High"].astype(float) for d, grp in intra.groupby("__date")}

    sold = False
    sell_dt_et: Optional[str] = None
    sell_price: Optional[float] = None
    days_to_sell: Optional[int] = None

    sim_days = list(sim_daily["__date"].tolist())
    _log(f"[SIM] Simulation started. Looping through {len(sim_days) - 1} trading days...")

    for day_idx in range(1, len(sim_days)):
        d = sim_days[day_idx]
        A = avg_cost
        L_avg = A
        L_big = A * (1.0 + params.sell_r)
        target = L_big

        _log(f"[DAY {day_idx}/{len(sim_days)-1}] {d.isoformat()} | shares={shares}, avg_cost={_fmt_money(A)} | sell_target={_fmt_money(target)}")

        highs = highs_by_date.get(d)
        if highs is None or highs.empty:
            _log(f"[SELL] No intraday data for {d.isoformat()}, cannot check for sell. Skipping.")
        else:
            day_high = float(highs.max())
            if day_high >= target:
                sold = True
                sell_price = target
                sell_dt_et = f"{d.isoformat()}"
                days_to_sell = (d - first_trade_date).days
                proceeds = shares * sell_price
                profit = proceeds - total_cost
                _log(f"✅ [SELL] Triggered: day_high={_fmt_money(day_high)} >= target={_fmt_money(target)}. Sold ALL {shares} shares at {_fmt_money(sell_price)}.")
                _log("[SIM] Finished.")
                return BacktestResult(
                    sold=True,
                    sell_dt_et=sell_dt_et,
                    sell_price=sell_price,
                    days_to_sell=days_to_sell,
                    ticker=ticker,
                    first_buy_dt_et=first_buy_dt_et,
                    first_buy_price=first_open,
                    final_shares=shares,
                    final_avg_cost=avg_cost,
                    total_cost_usd=total_cost,
                    proceeds_usd=proceeds,
                    profit_usd=profit,
                    profit_pct_on_cost=(profit / total_cost) if total_cost > 0 else None,
                    proceeds_krw=proceeds * params.fx_krw_per_usd,
                    profit_krw=profit * params.fx_krw_per_usd,
                    logs=logs,
                )
            else:
                _log(f"[SELL] Not triggered: day_high={_fmt_money(day_high)} < target={_fmt_money(target)}.")

        day_row = sim_daily[sim_daily["__date"] == d]
        if day_row.empty:
            _log(f"[BUY] No daily bar for {d.isoformat()}. Skipping buys.")
            continue
        C = float(day_row.iloc[0]["Close"])

        X = params.daily_budget_usd
        pair_cost = L_avg + L_big
        n_pair = int(X // pair_cost) if pair_cost > 0 else 0
        rem = X - n_pair * pair_cost
        ord_avg = n_pair
        ord_big = n_pair

        if params.prefer_avg_buy:
            if rem >= L_avg:
                ord_avg += 1
        else:
            if rem >= L_big:
                ord_big += 1
        _log(f"[ORDER] Reservation: budget=${_fmt_money(X)}, ord_avg={ord_avg}, ord_big={ord_big}.")

        fill_avg, fill_big = 0, 0
        if C <= A:
            fill_avg, fill_big = ord_avg, ord_big
        elif A < C <= L_big:
            fill_avg, fill_big = 0, ord_big
        
        buy_shares = fill_avg + fill_big
        if buy_shares == 0:
            _log(f"[BUY] No fills based on Close={_fmt_money(C)}.")
            continue

        buy_cost = buy_shares * C
        _log(f"[BUY] Filled: {buy_shares} shares at Close={_fmt_money(C)}, cost=${_fmt_money(buy_cost)}.")
        
        shares += buy_shares
        total_cost += buy_cost
        avg_cost = total_cost / shares
        _log(f"[POS] New position: shares={shares}, avg_cost=${_fmt_money(avg_cost)}, total_cost=${_fmt_money(total_cost)}.")

    _log("\n[END] Sell target was not reached within the available history. Simulation ended without a sell event.")
    _log("[SIM] Finished.")
    return BacktestResult(
        sold=False,
        sell_dt_et=None,
        sell_price=None,
        days_to_sell=None,
        ticker=ticker,
        first_buy_dt_et=first_buy_dt_et,
        first_buy_price=first_open,
        final_shares=shares,
        final_avg_cost=avg_cost,
        total_cost_usd=total_cost,
        proceeds_usd=None,
        profit_usd=None,
        profit_pct_on_cost=None,
        proceeds_krw=None,
        profit_krw=None,
        logs=logs,
    )
