from __future__ import annotations

import asyncio
import os
import logging
import queue
import time
from datetime import datetime
from typing import Deque, List, Optional, Set

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter, Retry
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from backtest import BacktestParams, BacktestResult, run_backtest


# -------------------------
# Logging
# -------------------------
# Configure root logger for stdout
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
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


# -------------------------
# Networking (for yfinance)
# -------------------------
# NOTE: Removed custom requests.Session as yfinance seems to handle its own session
# and explicitly providing one can lead to YFDataException errors with curl_cffi.


# -------------------------
# Data fetchers (yfinance)
# -------------------------
def fetch_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


def fetch_intraday(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        prepost=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


# -------------------------
# Telegram helpers
# -------------------------
def _ensure_allowed(update: Update, allowed_ids: Set[int]) -> bool:
    chat_id = update.effective_chat.id if update.effective_chat else None
    if (chat_id is None) or (chat_id not in allowed_ids):
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
    lines = [
        f"*Ticker:* `{res.ticker}`",
        f"*First buy:* {res.first_buy_dt_et} @ `${res.first_buy_price:,.2f}`",
        f"*Final shares:* `{res.final_shares}`",
        f"*Final avg cost:* `${res.final_avg_cost:,.2f}`",
        f"*Total cost:* `${res.total_cost_usd:,.2f}`",
    ]
    if res.sold:
        lines.extend([
            "",
            "✅ *SOLD triggered*",
            f"*Sell time:* {res.sell_dt_et}",
            f"*Sell price:* `${res.sell_price:,.2f}`",
            f"*Days to sell:* `{res.days_to_sell}` days",
            f"*Proceeds:* `${res.proceeds_usd:,.2f}`",
            f"*Profit:* `${res.profit_usd:,.2f}` ({res.profit_pct_on_cost*100:.2f}%)",
            f"*Proceeds (KRW):* `₩{res.proceeds_krw:,.0f}`",
            f"*Profit (KRW):* `₩{res.profit_krw:,.0f}`",
        ])
    else:
        lines.extend([
            "",
            "⚠️ *Sell target not reached within available history.*",
        ])
    return "\n".join(lines)


# -------------------------
# Command handlers
# -------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _ensure_allowed(update, context.bot_data["allowed_chat_ids"]):
        return
    defaults = context.bot_data["defaults"]
    msg = (
        "DCA Backtester Bot is running.\n\n"
        "Use /ping to check status and defaults.\n"
        "Use /backtest to run a simulation.\n\n"
        "Example:\n"
        "`/backtest QQQ 2023-01-01 500 true`"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    is_allowed = _ensure_allowed(update, context.bot_data["allowed_chat_ids"])
    
    defaults = context.bot_data["defaults"]
    lines = [
        "*Bot Status:* `ONLINE`",
        f"*Chat ID:* `{chat_id}`",
        f"*Authorized:* {'`YES`' if is_allowed else '`NO`'}",
        "",
        "*Current Defaults:*",
        f"  `DEFAULT_SELL_R` = `{defaults['sell_r']}`",
        f"  `DEFAULT_FX` = `{defaults['fx']}`",
        f"  `DEFAULT_INTRADAY_INTERVAL` = `{defaults['interval']}`",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def telegram_logger_task(
    update: Update, log_queue: queue.Queue, stop_event: asyncio.Event
):
    """Async task to poll a queue for logs and send them to Telegram."""
    log_buffer: List[str] = []
    last_send_time = time.time()

    async def send_logs():
        nonlocal last_send_time
        if not log_buffer:
            return
        
        message = "\n".join(log_buffer)
        log_buffer.clear()
        
        # Use a code block for readability
        for chunk in _chunk_text(f"```\n{message}\n```"):
            try:
                await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN_V2)
            except Exception as e:
                logger.error(f"Failed to send log chunk to Telegram: {e}")
        last_send_time = time.time()

    while not stop_event.is_set():
        try:
            # Poll queue without blocking the event loop
            log_entry = await asyncio.to_thread(log_queue.get, timeout=0.1)
            log_buffer.append(log_entry)
            log_queue.task_done()
        except queue.Empty:
            # If queue is empty, check if it's time to send buffered logs
            if time.time() - last_send_time > 2.0 and log_buffer:
                await send_logs()
            continue # Poll again
        
        # Send if buffer is full
        if len(log_buffer) >= 20:
            await send_logs()

    # Send any remaining logs after stop signal
    await send_logs()


async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _ensure_allowed(update, context.bot_data["allowed_chat_ids"]):
        return

    args = context.args
    defaults = context.bot_data["defaults"]
    log_history = Deque[str](maxlen=20) # For error reporting

    # --- 1. Immediate ACK and Argument Parsing ---
    try:
        if len(args) < 3:
            await update.message.reply_text(
                "Usage: `/backtest <TICKER> <YYYY-MM-DD> <DAILY_BUDGET> [prefer_avg_buy] [sell_r] [fx] [interval]`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        ticker = args[0].strip().upper()
        start_date = args[1].strip()
        datetime.strptime(start_date, "%Y-%m-%d") # Validate format
        daily_budget = float(args[2])

        prefer_avg_buy = _to_bool(args[3]) if len(args) > 3 else True
        sell_r = float(args[4]) if len(args) > 4 else defaults["sell_r"]
        fx = float(args[5]) if len(args) > 5 else defaults["fx"]
        interval = args[6].strip() if len(args) > 6 else defaults["interval"]

        params = BacktestParams(
            ticker=ticker,
            start_date=start_date,
            daily_budget_usd=daily_budget,
            sell_r=sell_r,
            fx_krw_per_usd=fx,
            prefer_avg_buy=prefer_avg_buy,
            intraday_interval=interval,
        )
        
        ack_msg = (
            f"✅ *Backtest accepted for `{ticker}`*\n\n"
            f"*Start Date:* {start_date}\n"
            f"*Daily Budget:* ${daily_budget:,.2f}\n"
            f"*Sell Target Ratio:* {sell_r}\n"
            f"*FX Rate:* {fx}\n"
            f"*Interval:* {interval}\n\n"
            "Starting process now. Logs will be streamed below."
        )
        await update.message.reply_text(ack_msg, parse_mode=ParseMode.MARKDOWN)

    except (ValueError, IndexError) as e:
        await update.message.reply_text(f"❌ *Argument Error:*\n`{e}`\nPlease check your inputs and try again.", parse_mode=ParseMode.MARKDOWN)
        return

    # --- 2. Setup Threaded Execution & Logging ---
    log_queue = queue.Queue()
    stop_event = asyncio.Event()

    def log_to_queue(msg: str):
        logger.info(msg) # Also print to stdout
        log_history.append(msg)
        log_queue.put_nowait(msg)

    logger_task = asyncio.create_task(
        telegram_logger_task(update, log_queue, stop_event)
    )

    # --- 3. Run Backtest in Thread ---
    try:
        log_to_queue("[BOT] Handler invoked, offloading to background thread.")
        
        result: BacktestResult = await asyncio.to_thread(
            run_backtest,
            params=params,
            fetch_daily=fetch_daily,
            fetch_intraday=fetch_intraday,
            log=log_to_queue,
        )
        
        log_to_queue("[BOT] Backtest finished successfully.")
        summary_md = _format_result_md(result)
        await update.message.reply_text(summary_md, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.exception("Backtest failed in background thread")
        error_msg = (
            f"❌ *Backtest failed:*\n`{type(e).__name__}: {e}`\n\n"
            "*Last few log messages:*\n"
            "```\n"
            f"{'\n'.join(log_history)}\n"
            "```"
        )
        for chunk in _chunk_text(error_msg):
             await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
    
    finally:
        # --- 4. Cleanup ---
        log_to_queue("[BOT] Task complete. Cleaning up.")
        # Signal the logger to stop and wait for it to finish sending logs
        stop_event.set()
        await logger_task


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
    logger.info(f"Defaults set: {defaults}")
    logger.info(f"Allowed chat IDs: {allowed}")

    app = Application.builder().token(token).build()
    app.bot_data["allowed_chat_ids"] = allowed
    app.bot_data["defaults"] = defaults

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))

    logger.info("Bot started and handlers are registered. Starting polling...")
    app.run_polling()
    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
