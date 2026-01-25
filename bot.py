from __future__ import annotations

import asyncio
import os
import logging
import time
from collections import deque
from datetime import datetime, timedelta
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




def _to_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: '{s}'")


def _get_intraday_start_date() -> str:
    """Calculates the recommended start date for intraday backtests (60 days ago)."""
    return (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")


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
        f"*Sell time:* {res.sell_dt_et} ({res.days_to_sell} days)",
        f"*Sell price:* ${res.sell_price:,.2f}",
        f"*Proceeds:* ${res.proceeds_usd:,.2f} (₩{res.proceeds_krw:,.0f})",
        f"*Profit({res.profit_pct_on_cost*100:.2f}%):* ${res.profit_usd:,.2f} (₩{res.profit_krw:,.0f})",
    ]
    return "\n".join(lines)


# -------------------------
# Command handlers
# -------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Use the intraday_start_date from bot_data
    intraday_start_date = context.bot_data.get("intraday_start_date", "YYYY-MM-DD")
    msg = (
        "DCA Backtester Bot is running.\n\n"
        "Use /ping to check status and defaults.\n"
        "Use /backtest to run a simulation.\n\n"
        "*Example:*\n"
        f"`/backtest TQQQ {intraday_start_date} 500`"
    )
    await update.effective_message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id

    defaults = context.bot_data["defaults"]
    lines = [
        "*Bot Status:* ONLINE",
        f"*Chat ID:* {chat_id}",
        "",
        "*Current Defaults:*",
        f"  Sell Ratio = {defaults['sell_r']}",
        f"  FX Rate = {defaults['fx']}",
        f"  Intraday Interval = {defaults['interval']}",
    ]
    await update.effective_message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    intraday_start_date = context.bot_data.get("intraday_start_date", "YYYY-MM-DD")
    
    help_message = (
        "Hello! I am a DCA Backtesting Bot.\n\n"
        "*Available Commands:*\n"
        "  /start - Get a welcome message and basic bot information.\n"
        "  /help - Display this help message.\n"
        "  /ping - Check bot status and current default settings.\n"
        "  /backtest - Run a backtest simulation.\n\n"
        "*Usage for /backtest:*\n"
        "  */backtest <TICKER> <YYYY-MM-DD> <DAILY_BUDGET> [prefer_avg_buy] [sell_r] [fx] [interval]*\n\n"
        "*Arguments:*\n"
        "  *<TICKER>*: Stock ticker symbol (e.g., QQQ, SPY).\n"
        "  *<YYYY-MM-DD>*: Start date for the backtest (e.g., 2023-01-01).\n"
        "  *<DAILY_BUDGET>*: Daily budget in USD for purchases (e.g., 500).\n"
        "  *[prefer_avg_buy]*: Optional. *true* or *false*. Default is *true*.\n"
        "  *[sell_r]*: Optional. Sell target ratio (e.g., 0.10 for 10%). Default from config.\n"
        "  *[fx]*: Optional. FX rate KRW per USD (e.g., 1450). Default from config.\n"
        "  *[interval]*: Optional. Intraday interval (e.g., 5m, 15m). Default from config.\n\n"
        f"*Example:*\n"
        f"`/backtest TQQQ {intraday_start_date} 500`\n\n"
    )
    await update.effective_message.reply_text(help_message, parse_mode=ParseMode.MARKDOWN)


async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    defaults = context.bot_data["defaults"]
    log_history: Deque[str] = deque(maxlen=20)

    # --- 1. Immediate ACK and Argument Parsing ---
    try:
        if len(args) < 3:
            await update.effective_message.reply_text(
                "Usage: */backtest <TICKER> <YYYY-MM-DD> <DAILY_BUDGET> [prefer_avg_buy] [sell_r] [fx] [interval]*",
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
        await update.effective_message.reply_text(ack_msg, parse_mode=ParseMode.MARKDOWN)

    except (ValueError, IndexError) as e:
        await update.effective_message.reply_text(f"❌ *Argument Error:*\n{e}\nPlease check your inputs.", parse_mode=ParseMode.MARKDOWN)
        return

    # --- 2. Setup File-based Logging ---
    file_handler = None
    run_logger = logging.getLogger(f"run_{ticker}_{time.time_ns()}")
    run_logger.propagate = False # Prevent duplicate stdout logs
    
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        # Use one log file per day
        log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

        # Append to the daily log file
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s"))
        run_logger.addHandler(file_handler)
        run_logger.setLevel(logging.INFO)

        def log_to_file_and_history(msg: str):
            run_logger.info(msg)
            log_history.append(msg)

        # --- 3. Run Backtest in Thread ---
        log_to_file_and_history(f"--- Starting backtest for {ticker} from chat {update.effective_chat.id} ---")

        result: BacktestResult = await asyncio.to_thread(
            run_backtest,
            params=params,
            fetch_daily=fetch_daily,
            fetch_intraday=fetch_intraday,
            log=log_to_file_and_history,
        )

        log_to_file_and_history(f"[BOT] Backtest for {ticker} finished successfully.")
        summary_md = _format_result_md(result)
        await update.effective_message.reply_text(summary_md, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.exception(f"Backtest for {ticker} failed in background thread")
        log_to_file_and_history(f"[BOT] Backtest for {ticker} failed with exception: {e}")
        
        error_msg = (
            f"❌ *Backtest for {ticker} failed:*\n{type(e).__name__}: {e}\n\n"
            "*Last few log messages:*\n"
            "```\n"
            f"{'\n'.join(log_history)}\n"
            "```"
        )
        for chunk in _chunk_text(error_msg):
            await update.effective_message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    finally:
        # --- 4. Cleanup ---
        if file_handler:
            log_to_file_and_history(f"--- Backtest for {ticker} complete ---")
            file_handler.close()
            run_logger.removeHandler(file_handler)


def main() -> None:
    load_dotenv()
    logger.info("Env vars loaded from .env")

    token = _get_env_required("TELEGRAM_BOT_TOKEN")
    
    defaults = {
        "fx": float(os.getenv("DEFAULT_FX", "1450.0")),
        "sell_r": float(os.getenv("DEFAULT_SELL_R", "0.10")),
        "interval": os.getenv("DEFAULT_INTRADAY_INTERVAL", "5m"),
    }

    # Fetch oldest date for help message and store in bot_data
    intraday_start_date = _get_intraday_start_date()
    logger.info(f"Calculated recommended intraday start date: {intraday_start_date}")

    app = Application.builder().token(token).build()
    # No longer setting allowed_chat_ids in bot_data
    app.bot_data["defaults"] = defaults
    app.bot_data["intraday_start_date"] = intraday_start_date

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))

    logger.info("Bot started and handlers are registered. Starting polling...")
    app.run_polling()
    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
