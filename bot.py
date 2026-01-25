from __future__ import annotations

import asyncio
import os
import logging
import time
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional, Set

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from backtest import BacktestParams, BacktestResult, run_backtest

# -------------------------
# Global & App-level Logging
# -------------------------
# Configure root logger for stdout
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("dca-backtester-bot")


# -------------------------
# Env / Config
# -------------------------
def _get_env_required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _parse_allowed_chat_ids(s: str) -> Set[int]:
    out: Set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _to_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: '{s}'")


def _get_oldest_yf_date(ticker: str = "SPY") -> str:
    """Fetches the oldest available date for a given ticker from Yahoo Finance."""
    try:
        hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
        if not hist.empty:
            return hist.index[0].strftime("%Y-%m-%d")
    except Exception as e:
        logger.error(f"Could not fetch oldest date for {ticker}: {e}")
    return "1993-01-29" # Fallback to SPY's known start date


# -------------------------
# Data fetchers (yfinance)
# -------------------------
def fetch_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker, start=start, end=end, interval="1d",
        auto_adjust=False, actions=False, progress=False, group_by="column",
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def fetch_intraday(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker, start=start, end=end, interval=interval,
        auto_adjust=False, actions=False, progress=False, group_by="column", prepost=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


# -------------------------
# Telegram helpers
# -------------------------
def _ensure_allowed(update: Update, allowed_ids: Set[int]) -> bool:
    chat_id = update.effective_chat.id
    if not chat_id or chat_id not in allowed_ids:
        logger.warning(f"Rejected access from chat_id={chat_id}")
        return False
    return True


def _chunk_text(s: str, limit: int = 4000) -> List[str]:
    chunks: List[str] = []
    buf = ""
    for line in s.splitlines(True):
        if len(buf) + len(line) > limit:
            chunks.append(buf)
            buf = ""
        buf += line
    if buf:
        chunks.append(buf)
    return chunks


def _format_result_md(res: BacktestResult) -> str:
    # Only return the single "not reached" line if not sold.
    if not res.sold:
        return "⚠️ *Sell target not reached within available history.*"

    lines = [
        f"*Ticker:* {res.ticker}",
        f"*First buy:* {res.first_buy_dt_et} @ ${res.first_buy_price:,.2f}",
        f"*Final shares:* {res.final_shares}",
        f"*Final avg cost:* ${res.final_avg_cost:,.2f}",
        f"*Total cost:* ${res.total_cost_usd:,.2f}",
        "",
        "✅ *SOLD triggered*",
        f"*Sell time:* {res.sell_dt_et}",
        f"*Sell price:* ${res.sell_price:,.2f}",
        f"*Days to sell:* {res.days_to_sell} days",
        f"*Proceeds:* ${res.proceeds_usd:,.2f}",
        f"*Profit:* ${res.profit_usd:,.2f} ({res.profit_pct_on_cost*100:.2f}%)",
        f"*Proceeds (KRW):* ₩{res.proceeds_krw:,.0f}",
        f"*Profit (KRW):* ₩{res.profit_krw:,.0f}",
    ]
    return "\n".join(lines)


# -------------------------
# Command handlers
# -------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _ensure_allowed(update, context.bot_data["allowed_chat_ids"]):
        return

    oldest_date = context.bot_data.get("oldest_yf_date", "1993-01-29")
    msg = (
        "DCA Backtester Bot is running.\n\n"
        "Use /ping to check status and defaults.\n"
        "Use /backtest to run a simulation.\n\n"
        "*Example:*\n"
        f"`/backtest QQQ {oldest_date} 500 true`"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    is_allowed = _ensure_allowed(update, context.bot_data["allowed_chat_ids"])

    defaults = context.bot_data["defaults"]
    lines = [
        "*Bot Status:* ONLINE",
        f"*Chat ID:* {chat_id}",
        f"*Authorized:* {'YES' if is_allowed else 'NO'}",
        "",
        "*Current Defaults:*",
        f"  Sell Ratio = {defaults['sell_r']}",
        f"  FX Rate = {defaults['fx']}",
        f"  Intraday Interval = {defaults['interval']}",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _ensure_allowed(update, context.bot_data["allowed_chat_ids"]):
        return

    args = context.args
    defaults = context.bot_data["defaults"]
    log_history: Deque[str] = deque(maxlen=20)

    # --- 1. Immediate ACK and Argument Parsing ---
    try:
        if len(args) < 3:
            await update.message.reply_text(
                "Usage: `/backtest <TICKER> <YYYY-MM-DD> <DAILY_BUDGET> [prefer_avg_buy] [sell_r] [fx] [interval]`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        ticker = args[0].strip().upper()
        start_date_str = args[1].strip()
        datetime.strptime(start_date_str, "%Y-%m-%d")  # Validate format
        daily_budget = float(args[2])

        prefer_avg_buy = _to_bool(args[3]) if len(args) > 3 else True
        sell_r = float(args[4]) if len(args) > 4 else defaults["sell_r"]
        fx = float(args[5]) if len(args) > 5 else defaults["fx"]
        interval = args[6].strip() if len(args) > 6 else defaults["interval"]

        params = BacktestParams(
            ticker=ticker, start_date=start_date_str, daily_budget_usd=daily_budget,
            sell_r=sell_r, fx_krw_per_usd=fx, prefer_avg_buy=prefer_avg_buy, intraday_interval=interval,
        )

        ack_msg = (
            f"✅ *Backtest accepted for {ticker}*\n\n"
            f"Parameters:\n"
            f"  Start Date: {start_date_str}\n"
            f"  Daily Budget: ${daily_budget:,.2f}\n"
            f"  Sell Target: {sell_r}\n\n"
            "Process starting now. A summary will be sent on completion, or an error if it fails."
        )
        await update.message.reply_text(ack_msg, parse_mode=ParseMode.MARKDOWN)

    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"❌ *Argument Error:*\n{e}\nPlease check your inputs.", parse_mode=ParseMode.MARKDOWN)
        return

    # --- 2. Setup File-based Logging ---
    file_handler = None
    run_logger = logging.getLogger(f"run_{ticker}_{time.time_ns()}")
    run_logger.propagate = False # Prevent duplicate stdout logs
    
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{ticker}_{timestamp}.log")

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        run_logger.addHandler(file_handler)
        run_logger.setLevel(logging.INFO)

        def log_to_file_and_history(msg: str):
            run_logger.info(msg)
            log_history.append(msg)

        # --- 3. Run Backtest in Thread ---
        log_to_file_and_history("[BOT] Handler invoked, offloading to background thread.")

        result: BacktestResult = await asyncio.to_thread(
            run_backtest,
            params=params,
            fetch_daily=fetch_daily,
            fetch_intraday=fetch_intraday,
            log=log_to_file_and_history,
        )

        log_to_file_and_history(f"[BOT] Backtest finished. Log file: {log_file}")
        summary_md = _format_result_md(result)
        await update.message.reply_text(summary_md, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.exception("Backtest failed in background thread")
        log_to_file_and_history(f"[BOT] Backtest failed with exception: {e}")
        
        error_msg = (
            f"❌ *Backtest for {ticker} failed:*\n{type(e).__name__}: {e}\n\n"
            "*Last few log messages:*\n"
            "```\n"
            f"{'\n'.join(log_history)}\n"
            "```"
        )
        for chunk in _chunk_text(error_msg):
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    finally:
        # --- 4. Cleanup ---
        if file_handler:
            file_handler.close()
            run_logger.removeHandler(file_handler)


def main() -> None:
    load_dotenv()
    logger.info("Env vars loaded from .env")

    token = _get_env_required("TELEGRAM_BOT_TOKEN")
    allowed_ids_str = _get_env_required("TELEGRAM_ALLOWED_CHAT_IDS")
    allowed = _parse_allowed_chat_ids(allowed_ids_str)
    
    defaults = {
        "fx": float(os.getenv("DEFAULT_FX", "1450.0")),
        "sell_r": float(os.getenv("DEFAULT_SELL_R", "0.10")),
        "interval": os.getenv("DEFAULT_INTRADAY_INTERVAL", "5m"),
    }

    # Fetch oldest date for help message and store in bot_data
    oldest_date = _get_oldest_yf_date()
    logger.info(f"Fetched oldest available YF date: {oldest_date}")

    app = Application.builder().token(token).build()
    app.bot_data["allowed_chat_ids"] = allowed
    app.bot_data["defaults"] = defaults
    app.bot_data["oldest_yf_date"] = oldest_date

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))

    logger.info("Bot started and handlers are registered. Starting polling...")
    app.run_polling()
    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
