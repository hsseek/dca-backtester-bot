from __future__ import annotations

import atexit
import fcntl
import asyncio
import html
import json
import logging
import os
import time
import traceback
from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pytz
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

LOCK_FILE = "logs/bot.lock"


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



US_MARKET_TZ = ZoneInfo("America/New_York")
PRE_MARKET_START_HOUR_ET = 4  # 4:00 AM ET
AFTER_MARKET_END_HOUR_ET = 20  # 8:00 PM ET

_last_alert_times: dict[str, datetime] = {}
async def monitor_prices(application: Application) -> None:
    bot_data = application.bot_data
    monitored_tickers = bot_data.get("monitored_tickers", [])
    monitoring_interval_seconds = bot_data.get("monitoring_interval_seconds", 3600)
    admin_user_id = bot_data.get("admin_user_id")

    if not monitored_tickers:
        logger.info("No tickers configured for monitoring. Monitoring task will not start.")
        return

    logger.info(f"Starting price monitoring for {len(monitored_tickers)} tickers "
                f"every {monitoring_interval_seconds} seconds.")

    while True:
        now_et = datetime.now(US_MARKET_TZ)
        current_hour_et = now_et.hour

        # Check if within US trading hours (4 AM ET to 8 PM ET)
        if PRE_MARKET_START_HOUR_ET <= current_hour_et < AFTER_MARKET_END_HOUR_ET:
            logger.info("Running scheduled price monitoring check within US trading hours.")
            for ticker, threshold_days in monitored_tickers:
                try:
                    # Calculate start date for historical data
                    # Fetch enough data to ensure we get 'threshold_days' *trading days*.
                    # A buffer of +5 days is added to account for weekends/holidays.
                    end_date = now_et.date()
                    start_date = end_date - timedelta(days=threshold_days + 10) # Increased buffer

                    df = await fetch_daily(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

                    if df.empty:
                        logger.warning(f"No data fetched for {ticker} for monitoring. Skipping.")
                        continue

                    # Ensure we have enough data points: at least threshold_days + 1 for comparison
                    # (threshold_days for historical range + 1 for current price)
                    if len(df) < threshold_days + 1:
                        logger.info(f"Not enough total trading days data ({len(df)}) for {ticker}. Need at least {threshold_days + 1}. Skipping.")
                        continue

                    # The most recent close is our 'current_price'
                    current_price = df["Close"].iloc[-1]
                    current_price_date = df.index[-1].date()

                    # The historical data for comparison is the 'threshold_days' trading days *before* the current_price_date
                    historical_df_for_comparison = df.iloc[-(threshold_days + 1):-1]
                    
                    # This should always have 'threshold_days' entries if len(df) was sufficient
                    if len(historical_df_for_comparison) < threshold_days:
                         logger.warning(f"Unexpected: historical_df_for_comparison has only {len(historical_df_for_comparison)} entries, expected {threshold_days}. Skipping.")
                         continue

                    min_price_series = historical_df_for_comparison["Close"]
                    lowest_historical_price = min_price_series.min()
                    lowest_historical_price_date = min_price_series.idxmin().date()


                    # Check if current price is the lowest in the historical period
                    if current_price < lowest_historical_price:
                        # Implement a cool-down to prevent spamming
                        last_alert_time = _last_alert_times.get(ticker)
                        if not last_alert_time or (datetime.now() - last_alert_time) > timedelta(hours=24): # 24-hour cooldown
                            message = f"🔔 *Buy now!* 🔔\n" \
                                      f"*{ticker}* is at its lowest price in the last *{threshold_days} trading days*!\n" \
                                      f"Current Price: *${current_price:,.2f}* (as of {current_price_date})\n" \
                                      f"Lowest {threshold_days}-day historical trading price: *${lowest_historical_price:,.2f}* (on {lowest_historical_price_date})"
                            if admin_user_id:
                                try:
                                    await application.bot.send_message(chat_id=admin_user_id, text=message, parse_mode=ParseMode.MARKDOWN)
                                    logger.info(f"Sent 'Buy now!' alert for {ticker} to admin {admin_user_id}.")
                                    _last_alert_times[ticker] = datetime.now()
                                except Exception as e:
                                    logger.error(f"Failed to send 'Buy now!' alert for {ticker} to admin {admin_user_id}: {e}")
                            else:
                                logger.warning(f"Admin user ID not configured. Could not send 'Buy now!' alert for {ticker}.")
                    else:
                        logger.info(f"{ticker} (Current: ${current_price:,.2f}) not at {threshold_days}-day trading low (Lowest: ${lowest_historical_price:,.2f} on {lowest_historical_price_date}).")

                except Exception as e:
                    logger.error(f"Error during price monitoring for {ticker}: {e}", exc_info=True)
                    if admin_user_id:
                        error_message = (
                            f"❌ *Price Monitoring Error for {ticker}:*\n"
                            f"An unexpected error occurred: `{type(e).__name__}: {e}`\n"
                            f"Please check bot logs for details."
                        )
                        try:
                            await application.bot.send_message(chat_id=admin_user_id, text=error_message, parse_mode=ParseMode.MARKDOWN)
                        except Exception as send_e:
                            logger.error(f"Failed to send error notification to admin {admin_user_id}: {send_e}")
            
            # Sleep for the configured interval if within trading hours
            await asyncio.sleep(monitoring_interval_seconds)

        else:
            # Outside trading hours, calculate sleep until next 4 AM ET
            logger.info("Outside US trading hours. Sleeping until next trading window.")
            next_4am_et = now_et.replace(hour=PRE_MARKET_START_HOUR_ET, minute=0, second=0, microsecond=0)
            if now_et.hour >= AFTER_MARKET_END_HOUR_ET:  # If past 8 PM ET, sleep until 4 AM next day
                next_4am_et += timedelta(days=1)
            
            sleep_duration = (next_4am_et - now_et).total_seconds()
            if sleep_duration < 0: # Should not happen if logic is correct, but defensive
                sleep_duration = 0
            logger.info(f"Sleeping for {sleep_duration:.0f} seconds until {next_4am_et.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")
            await asyncio.sleep(sleep_duration)

async def daily_report_task(application: Application) -> None:
    bot_data = application.bot_data
    admin_user_id = bot_data.get("admin_user_id")
    monitored_tickers = bot_data.get("monitored_tickers", [])

    if not admin_user_id:
        logger.warning("Admin user ID not configured. Daily report task will not start.")
        return

    logger.info("Starting daily report task.")

    while True:
        now_et = datetime.now(US_MARKET_TZ)
        # Target time is 9:30 AM ET
        target_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        
        if now_et >= target_time:
            target_time += timedelta(days=1)
            
        sleep_seconds = (target_time - now_et).total_seconds()
        logger.info(f"Daily report task sleeping for {sleep_seconds:.0f} seconds until {target_time}.")
        await asyncio.sleep(sleep_seconds)
        
        # Now it's 9:30 AM ET
        logger.info("Executing daily report.")
        
        # Check if it's a trading day using SPY as a proxy if no tickers are monitored
        check_ticker = monitored_tickers[0][0] if monitored_tickers else "SPY"
        
        try:
            today_str = datetime.now(US_MARKET_TZ).strftime("%Y-%m-%d")
            # Wait a few seconds to ensure data is likely available on yfinance
            await asyncio.sleep(30)
            
            # Fetch data for today to check if market is open
            df_today = await fetch_daily(check_ticker, today_str, (datetime.now(US_MARKET_TZ) + timedelta(days=1)).strftime("%Y-%m-%d"))
            
            if df_today.empty or "Open" not in df_today.columns or pd.isna(df_today["Open"].iloc[-1]):
                # Non-trading day
                await application.bot.send_message(
                    chat_id=admin_user_id,
                    text=f"📅 *{today_str}* is a non-trading day.",
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                # Trading day
                if not monitored_tickers:
                    await application.bot.send_message(
                        chat_id=admin_user_id,
                        text=f"📊 *Daily Report - {today_str}*\nNo tickers configured for monitoring.",
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    report_lines = [f"📊 *Daily Report - {today_str}*"]
                    for ticker, threshold_days in monitored_tickers:
                        try:
                            # Fetch historical data including today
                            end_date = datetime.now(US_MARKET_TZ).date()
                            start_date = end_date - timedelta(days=threshold_days + 15)
                            
                            df = await fetch_daily(ticker, start_date.strftime("%Y-%m-%d"), (end_date + timedelta(days=1)).strftime("%Y-%m-%d"))
                            
                            if df.empty or len(df) < 2:
                                report_lines.append(f"\n*{ticker}*: Insufficient data.")
                                continue
                                
                            # The last row is today
                            today_open = df["Open"].iloc[-1]
                            # Historical is everything BEFORE today
                            historical_df = df.iloc[:-1].tail(threshold_days)
                            
                            if len(historical_df) == 0:
                                report_lines.append(f"\n*{ticker}*: No historical data for comparison.")
                                continue

                            lowest_hist = historical_df["Close"].min()
                            diff_pct = (today_open - lowest_hist) / lowest_hist
                            
                            status_emoji = "🟢" if today_open > lowest_hist else "🔴"
                            
                            report_lines.append(
                                f"\n*{ticker}* ({threshold_days} trading days)"
                                f"\n  Open: *${today_open:,.2f}*"
                                f"\n  Lowest Hist Close: *${lowest_hist:,.2f}*"
                                f"\n  Diff: *{diff_pct:+.2%}* {status_emoji}"
                            )
                        except Exception as ticker_e:
                            logger.error(f"Error generating report for {ticker}: {ticker_e}")
                            report_lines.append(f"\n*{ticker}*: Error fetching data.")
                    
                    await application.bot.send_message(
                        chat_id=admin_user_id,
                        text="\n".join(report_lines),
                        parse_mode=ParseMode.MARKDOWN
                    )
                
        except Exception as e:
            logger.error(f"Error in daily report task: {e}", exc_info=True)
            try:
                await application.bot.send_message(
                    chat_id=admin_user_id,
                    text=f"❌ *Daily Report Error:*\n`{type(e).__name__}: {e}`"
                )
            except:
                pass

# File lock mechanism
_lock_fd = None

def acquire_lock() -> None:
    global _lock_fd
    try:
        os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
        _lock_fd = open(LOCK_FILE, "w")
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logger.info(f"Acquired lock file: {LOCK_FILE}")
    except BlockingIOError:
        logger.error(f"Another instance of the bot is already running. Could not acquire lock file: {LOCK_FILE}")
        # Optionally, clean up lock file if it's stale, but for now, just exit.
        exit(1)
    except Exception as e:
        logger.exception(f"Failed to acquire lock file: {LOCK_FILE}")
        exit(1)

def release_lock() -> None:
    global _lock_fd
    if _lock_fd:
        fcntl.flock(_lock_fd, fcntl.LOCK_UN)
        _lock_fd.close()
        logger.info(f"Released lock file: {LOCK_FILE}")
        if os.exists(LOCK_FILE):
            os.remove(LOCK_FILE)

def main() -> None:
    acquire_lock()
    atexit.register(release_lock)
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

    async def post_init(application: Application) -> None:
        if application.bot_data.get("monitored_tickers"):
            logger.info("Scheduling price monitoring task.")
            asyncio.create_task(monitor_prices(application))
        else:
            logger.info("No tickers configured for monitoring, price monitoring task will not be scheduled.")
            
        if application.bot_data.get("daily_report_enabled"):
            logger.info("Scheduling daily report task.")
            asyncio.create_task(daily_report_task(application))

    app.post_init = post_init

    # No longer setting allowed_chat_ids in bot_data
    app.bot_data["defaults"] = defaults
    app.bot_data["intraday_start_date"] = intraday_start_date
    app.bot_data["admin_user_id"] = os.getenv("ADMIN_USER_ID")
    app.bot_data["daily_report_enabled"] = _to_bool(os.getenv("DAILY_REPORT_ENABLED", "false"))

    # Monitoring feature configuration
    monitored_tickers_str = os.getenv("TICKERS_MONITORED", "")
    monitored_tickers: List[tuple[str, int]] = []
    if monitored_tickers_str:
        for entry in monitored_tickers_str.split(","):
            try:
                ticker, threshold_days_str = entry.strip().split(":")
                threshold_days = int(threshold_days_str)
                if threshold_days <= 0:
                    raise ValueError("Threshold days must be positive.")
                monitored_tickers.append((ticker.upper(), threshold_days))
            except ValueError as e:
                logger.warning(f"Invalid TICKERS_MONITORED entry '{entry}': {e}. Skipping.")
    app.bot_data["monitored_tickers"] = monitored_tickers

    monitoring_interval_seconds = int(os.getenv("MONITORING_INTERVAL_SECONDS", "3600"))
    if monitoring_interval_seconds <= 0:
        logger.warning("MONITORING_INTERVAL_SECONDS must be positive. Defaulting to 3600.")
        monitoring_interval_seconds = 3600
    app.bot_data["monitoring_interval_seconds"] = monitoring_interval_seconds

    if monitored_tickers:
        logger.info(f"Monitoring enabled for: {monitored_tickers} every {monitoring_interval_seconds} seconds.")
    else:
        logger.info("No tickers configured for monitoring.")

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
