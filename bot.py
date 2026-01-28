from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import time
import traceback
from collections import deque
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import Deque, List

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from backtest import BacktestParams, BacktestSeriesResult, run_backtest

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

MAX_LOG_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
MAX_BACKUP_LOG_FILES = 5


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
async def fetch_daily(ticker: str, start: str, end: str, timeout: int = 60) -> pd.DataFrame:
    """Asynchronously fetches daily data with a timeout."""
    def _download():
        return yf.download(
            tickers=ticker, start=start, end=end, interval="1d",
            auto_adjust=False, actions=False, progress=False, group_by="column",
        )

    loop = asyncio.get_running_loop()
    try:
        # noinspection PyTypeChecker
        df = await asyncio.wait_for(
            loop.run_in_executor(None, _download),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"yfinance daily data download for {ticker} timed out after {timeout} seconds.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


async def fetch_intraday(ticker: str, start: str, end: str, interval: str, timeout: int = 120) -> pd.DataFrame:
    """Asynchronously fetches intraday data with a timeout."""
    def _download():
        return yf.download(
            tickers=ticker, start=start, end=end, interval=interval,
            auto_adjust=False, actions=False, progress=False, group_by="column", prepost=False,
        )

    loop = asyncio.get_running_loop()
    try:
        # noinspection PyTypeChecker
        df = await asyncio.wait_for(
            loop.run_in_executor(None, _download),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"yfinance intraday data download for {ticker} ({interval}) timed out after {timeout} seconds.")

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


def _format_series_result_md(series_res: BacktestSeriesResult) -> str:
    lines = []
    
    # --- 1. Summary Section ---
    total_profit_usd = sum(t.profit_usd for t in series_res.completed_trades if t.profit_usd is not None)
    total_profit_krw = sum(t.profit_krw for t in series_res.completed_trades if t.profit_krw is not None)
    num_trades = len(series_res.completed_trades)

    # Use ticker from the first trade if available, otherwise from open position, or a placeholder
    ticker = "N/A"
    if series_res.completed_trades:
        ticker = series_res.completed_trades[0].ticker
    elif series_res.open_position_result:
        ticker = series_res.open_position_result.ticker

    lines.append(f"*{ticker} Backtest Summary*")
    lines.append(f"*Trades completed:* {num_trades}")
    lines.append(f"*Total Profit:* ${total_profit_usd:,.2f} (₩{total_profit_krw:,.0f})")
    lines.append("-" * 20)

    # --- 2. Individual Trades ---
    if not series_res.completed_trades:
        lines.append("No sell targets were reached during the simulation period.")
    else:
        for i, res in enumerate(series_res.completed_trades):
            lines.append(f"✅ *Trade #{i+1}: SOLD triggered*")
            
            profit_pct_str = f" ({res.profit_pct_on_cost*100:.2f}%)" if res.profit_pct_on_cost is not None else ""
            
            lines.extend([
                f"  *Duration:* {res.days_to_sell} days",
                f"  *First buy:* {res.first_buy_dt_et}",
                f"  *Total cost:* ${res.total_cost_usd:,.2f}",
                f"  *Sell time:* {res.sell_dt_et}",
                f"  *Sell price:* ${res.sell_price:,.2f}",
                f"  *Proceeds:* ${res.proceeds_usd:,.2f}",
                f"  *Profit{profit_pct_str}:* ${res.profit_usd:,.2f} (₩{res.profit_krw:,.0f})",
                ""]) # empty line for spacing

    lines.append("-" * 20)
    
    # --- 3. Final Open Position ---
    if series_res.open_position_result:
        res = series_res.open_position_result
        lines.append("💼 *Final Unsold Position*")
        
        unrealized_pl = 0.0
        unrealized_pl_rate = 0.0
        if series_res.last_day_close_price is not None:
            unrealized_pl = (series_res.last_day_close_price - res.final_avg_cost) * res.final_shares
            if res.final_avg_cost > 0:
                unrealized_pl_rate = (series_res.last_day_close_price - res.final_avg_cost) / res.final_avg_cost
        
        lines.extend([
            f"  *Ticker:* {res.ticker}",
            f"  *First buy:* {res.first_buy_dt_et} @ ${res.first_buy_price:,.2f}",
            f"  *Final shares:* {res.final_shares}",
            f"  *Final avg cost:* ${res.final_avg_cost:,.2f}",
            f"  *Total cost:* ${res.total_cost_usd:,.2f}",
            f"  *Unrealized P/L:* ${-unrealized_pl:,.2f} ({-unrealized_pl_rate:.2%})",
        ])
    else:
        lines.append("No open position at the end of the simulation.")
        
    return "\n".join(lines)


# -------------------------
# Command & Error handlers
# -------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a telegram message to notify the user and admin."""
    logger.error("Exception while handling an update:", exc_info=context.error)

    # Optionally, notify the user that an error occurred
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("An unexpected error occurred. The developer has been notified.")

    # Send a detailed error report to the admin
    admin_user_id = context.bot_data.get("admin_user_id")
    if admin_user_id and context.error:
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)
        
        # Format the message
        update_str = json.dumps(update.to_dict(), indent=2) if isinstance(update, Update) else str(update)
        message = (
            f"An exception was raised while handling an update\n"
            f"<pre>update = {html.escape(update_str)}</pre>\n\n"
            f"<pre>context.chat_data = {html.escape(str(context.chat_data))}</pre>\n\n"
            f"<pre>context.user_data = {html.escape(str(context.user_data))}</pre>\n\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )
        
        # Split the message into chunks to avoid hitting Telegram's message length limit
        for chunk in _chunk_text(message, limit=4096):
            await context.bot.send_message(chat_id=admin_user_id, text=chunk, parse_mode=ParseMode.HTML)


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Use the intraday_start_date from bot_data
    msg = (
        "DCA Backtester Bot is running.\n\n"
        "Use /ping to check status and defaults.\n"
        "Use /bt to run a simulation.\n\n"
        f"`/bt TQQQ`"
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
        f"  Daily Budget = {defaults['daily_budget']}",
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
        "  /bt - Run a backtest simulation.\n\n"
        "*Usage for /bt:*\n"
        "  */bt <TICKER> [YYYY-MM-DD] [DAILY_BUDGET] [sell_r] [fx] [prefer_avg_buy] [interval]*\n\n"
        "*Arguments:*\n"
        "  *<TICKER>*: Required. Stock ticker symbol (e.g., QQQ, SPY).\n"
        "  *[YYYY-MM-DD]*: Optional. Start date for the backtest. Defaults to the oldest available date for the chosen interval.\n"
        "  *[DAILY_BUDGET]*: Optional. Daily budget in USD. Defaults to the value in config.\n"
        "  *[sell_r]*: Optional. Sell target ratio (e.g., 0.10 for 10%). Default from config.\n"
        "  *[fx]*: Optional. FX rate KRW per USD (e.g., 1450). Default from config.\n"
        "  *[prefer_avg_buy]*: Optional. *true* or *false*. Default is *true*.\n"
        "  *[interval]*: Optional. Intraday interval (e.g., 5m, 15m). Default from config.\n\n"
        f"*Example (all optional args):*\n"
        f"`/bt TQQQ {intraday_start_date} 500 0.10 1450 true 5m`\n\n"
        f"*Example (ticker only):*\n"
        f"`/bt TQQQ`\n\n"
    )
    await update.effective_message.reply_text(help_message, parse_mode=ParseMode.MARKDOWN)


async def async_backtest_runner(params: BacktestParams, log_history: Deque[str], chat_id: int) -> BacktestSeriesResult:
    """
    Asynchronous helper to run the backtest and handle file logging.
    """
    file_handler = None
    # Use a unique logger name for each run to ensure thread safety with handlers
    run_logger = logging.getLogger(f"run_{params.ticker}_{time.time_ns()}")
    run_logger.propagate = False  # Prevent duplicate logs to stdout

    log_to_file_and_history = None
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "backtest.log")

        file_handler = RotatingFileHandler(
            log_file,
            mode='a',
            maxBytes=MAX_LOG_SIZE_BYTES,
            backupCount=MAX_BACKUP_LOG_FILES,
            encoding='utf-8'
        )
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s"))
        run_logger.addHandler(file_handler)
        run_logger.setLevel(logging.INFO)

        def _log_to_file_and_history(msg: str):
            run_logger.info(msg)
            log_history.append(msg)
        
        log_to_file_and_history = _log_to_file_and_history

        log_to_file_and_history(f"--- Starting backtest for {params.ticker} from chat {chat_id} ---")

        result = await run_backtest(
            params=params,
            fetch_daily=fetch_daily,
            fetch_intraday=fetch_intraday,
            log=log_to_file_and_history,
        )

        log_to_file_and_history(f"[BOT] Backtest for {params.ticker} finished successfully.")
        return result

    except Exception as e:
        # Log the exception to the file and re-raise it so the async handler can catch it
        logger.exception(f"Backtest for {params.ticker} failed in background task")
        log_history.append(f"[BOT] Backtest for {params.ticker} failed with exception: {e}")
        raise
    finally:
        if file_handler and log_to_file_and_history:
            log_to_file_and_history(f"--- Backtest for {params.ticker} complete ---")
            file_handler.close()
            run_logger.removeHandler(file_handler)


async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Handler /bt invoked with args: {context.args} from chat: {update.effective_chat.id}")

    args: List[str] = context.args
    defaults = context.bot_data["defaults"]
    log_history: Deque[str] = deque(maxlen=20)

    # --- 1. Argument Parsing ---
    try:
        if not args:
            await update.effective_message.reply_text(
                "Usage: `/bt <TICKER> [YYYY-MM-DD] [DAILY_BUDGET] [optional_args...]`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        ticker = args[0].strip().upper()
        
        start_date_str = args[1].strip() if len(args) > 1 else context.bot_data.get("intraday_start_date")
        if start_date_str:
            try:
                datetime.strptime(start_date_str, "%Y-%m-%d")
            except ValueError:
                await update.effective_message.reply_text(
                    f"❌ *Invalid Date Format: `{start_date_str}`*\n\nPlease use the `YYYY-MM-DD` format (e.g., `2023-01-01`)."
                )
                return

        try:
            daily_budget = float(args[2]) if len(args) > 2 else defaults["daily_budget"]
        except (ValueError, IndexError):
            await update.effective_message.reply_text(
                f"❌ *Invalid Budget: `{args[2] if len(args) > 2 else ''}`*\n\nPlease provide a number for the daily budget (e.g., `500`)."
            )
            return

        sell_r = float(args[3]) if len(args) > 3 else defaults["sell_r"]
        fx = float(args[4]) if len(args) > 4 else defaults["fx"]
        prefer_avg_buy = _to_bool(args[5]) if len(args) > 5 else True
        interval = args[6].strip() if len(args) > 6 else defaults["interval"]

        params = BacktestParams(
            ticker=ticker, start_date=start_date_str, daily_budget_usd=daily_budget,
            sell_r=sell_r, fx_krw_per_usd=fx, prefer_avg_buy=prefer_avg_buy, intraday_interval=interval,
        )

    except Exception as e:
        logger.error(f"Failed during argument parsing for /bt: {e}")
        await update.effective_message.reply_text(
            f"❌ *Argument Error:*\n{e}\nPlease check your inputs and use /help for more info."
        )
        return

    # --- 2. Acknowledge and Run in Background ---
    try:
        await update.effective_message.reply_text(
            f"✅ Backtest accepted for {ticker}\n"
            "Process starting now. A summary will be sent on completion, or an error if it fails."
        )

        # The whole backtest (including data fetching) should not take more than a few minutes.
        # Let's set a generous timeout of 5 minutes.
        result = await asyncio.wait_for(
            async_backtest_runner(
                params=params,
                log_history=log_history,
                chat_id=update.effective_chat.id
            ),
            timeout=300.0
        )
        
        summary_md = _format_series_result_md(result)
        await update.effective_message.reply_text(summary_md, parse_mode=ParseMode.MARKDOWN)

    except asyncio.TimeoutError:
        error_msg = (
            f"❌ *Backtest for {params.ticker} timed out after 5 minutes.*\n\n"
            "This could be due to slow data downloads or a very long simulation period.\n\n"
            "*Last few log messages:*\n"
            "```\n"
            f"{'\n'.join(log_history)}\n"
            "```"
        )
        for chunk in _chunk_text(error_msg):
            await update.effective_message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        error_msg = (
            f"❌ *Backtest for {params.ticker} failed:*\n{type(e).__name__}: {e}\n\n"
            "*Last few log messages:*\n"
            "```\n"
            f"{'\n'.join(log_history)}\n"
            "```"
        )
        for chunk in _chunk_text(error_msg):
            await update.effective_message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)


def main() -> None:
    load_dotenv()
    logger.info("Env vars loaded from .env")

    token = _get_env_required("TELEGRAM_BOT_TOKEN")
    
    defaults = {
        "fx": float(os.getenv("DEFAULT_FX", "1450.0")),
        "sell_r": float(os.getenv("DEFAULT_SELL_R", "0.10")),
        "interval": os.getenv("DEFAULT_INTRADAY_INTERVAL", "5m"),
        "daily_budget": float(os.getenv("DEFAULT_DAILY_BUDGET", "500.0")),
    }

    # Fetch the oldest date for help message and store in bot_data
    intraday_start_date = _get_intraday_start_date()
    logger.info(f"Calculated recommended intraday start date: {intraday_start_date}")

    app = Application.builder().token(token).build()
    # No longer setting allowed_chat_ids in bot_data
    app.bot_data["defaults"] = defaults
    app.bot_data["intraday_start_date"] = intraday_start_date
    app.bot_data["admin_user_id"] = os.getenv("ADMIN_USER_ID")

    # --- Register handlers ---
    app.add_error_handler(error_handler)

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))
    app.add_handler(CommandHandler("bt", backtest_cmd))

    logger.info("Bot started and handlers are registered. Starting polling...")
    app.run_polling(drop_pending_updates=True)
    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
