# DCA Backtester Bot for Telegram

A Python-based Telegram bot that runs a dollar-cost averaging (DCA) backtest using `yfinance` data. It allows users to simulate a DCA strategy against historical stock data and see the potential outcomes directly in their Telegram chat.

## Getting Started

### Prerequisites

- Python 3.9+
- `pip` for package management

### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/hsseek/dca-backtester-bot
    cd dca-backtester-bot
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    ```
    *On Windows, use `venv\Scripts\activate`*

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    -   Copy the example `.env.example` file to a new `.env` file:
        ```sh
        cp .env.example .env
        ```
    -   Open the `.env` file and add your Telegram Bot Token.

## Usage

1.  **Run the bot:**
    ```sh
    python bot.py
    ```

2.  **Interact with the bot on Telegram:**

    -   `/start`: Get a welcome message.
    -   `/help`: Display a detailed help message with all commands and arguments.
    -   `/ping`: Check if the bot is online and view its current default settings.
    -   `/backtest`: Run a simulation.

    **Backtest Command Usage:**
    ```
    /backtest <TICKER> <YYYY-MM-DD> <DAILY_BUDGET> [prefer_avg_buy] [sell_r] [fx] [interval]
    ```
    -   `<TICKER>`: Stock ticker symbol (e.g., QQQ, SPY).
    -   `<YYYY-MM-DD>`: Start date for the backtest.
    -   `<DAILY_BUDGET>`: Daily budget in USD for purchases.
    -   `[prefer_avg_buy]`: (Optional) `true` or `false`. Defaults to `true`.
    -   `[sell_r]`: (Optional) Sell target ratio.
    -   `[fx]`: (Optional) FX rate for currency conversion.
    -   `[interval]`: (Optional) Intraday interval (e.g., 5m, 15m).

    **Example:**
    ```
    /backtest TQQQ 2023-01-01 500
    ```

## Configuration

The following environment variables can be configured in the `.env` file:

-   `TELEGRAM_BOT_TOKEN`: **(Required)** Your Bot token.
-   `DEFAULT_FX`: The default foreign exchange rate to use for currency conversions.
-   `DEFAULT_SELL_R`: The default sell target ratio (e.g., `0.10` for 10%).
-   `DEFAULT_INTRADAY_INTERVAL`: The default interval for fetching intraday data (e.g., `5m`, `15m`, `30m`).