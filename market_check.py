from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import math

import pandas as pd
import yfinance as yf


# -------------------------------
# Settings
# -------------------------------
RISK_PER_TRADE = 50.0  # Fixed £ risk per trade

# Price scale used for sizing and portfolio valuation
# 1.0  = use price as-is
# 0.01 = divide by 100 because Yahoo price is effectively in pence for cash calcs
PRICE_SCALE = {
    "VUAG": 1.0,
    "CUKX": 0.01,
    "EMIM": 0.01,
    "CU31": 0.01,
    "IGLT": 1.0,
    "GBUS": 1.0,
    "CMOD": 1.0,
    "SGLN": 1.0,
}


# -------------------------------
# Watchlist (Trading 212 tickers)
# -------------------------------
WATCHLIST: Dict[str, str] = {
    "VUAG": "VUAG.L",
    "CUKX": "CUKX.L",
    "EMIM": "EMIM.L",
    "CU31": "CU31.L",
    "IGLT": "IGLT.L",
    "GBUS": "GBUS.L",
    "CMOD": "CMOD.L",
    "SGLN": "SGLN.L",
}

# Tickers that should use CLOSE-based channels
CLOSE_ONLY = {"SGLN"}


@dataclass
class SignalResult:
    name: str
    symbol: str
    close: float
    prior_50_high: float
    prior_20_low: float
    pct_to_breakout: float
    pct_to_sale: float
    entry_trigger: bool
    exit_trigger: bool
    status: str
    pct_change_from_yesterday: float
    risk_per_share: float | None
    position_size: int | None
    capital_required: float | None


# -------------------------------
# Data download & cleaning
# -------------------------------

def download_ohlc(symbol: str, period: str = "18mo") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {symbol}: {missing}")

    df = df.dropna(subset=required).copy()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.tail(200)

    if len(df) < 60:
        raise ValueError(f"Not enough data for {symbol}")

    return df


# -------------------------------
# Symbol-specific fixes
# -------------------------------

def apply_symbol_fixes(name: str, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # GBUS Yahoo anomaly fix:
    # Sometimes only the latest close is shown in pence rather than dollars
    if symbol == "GBUS.L":
        last_close = df.iloc[-1]["Close"]
        recent_closes = df["Close"].iloc[-10:-1]

        if len(recent_closes) > 0:
            median_close = recent_closes.median()

            if median_close > 0 and last_close / median_close > 50:
                df.iloc[-1, df.columns.get_loc("Close")] = last_close / 100.0

    return df


# -------------------------------
# Signal calculation
# -------------------------------

def compute_channels(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if name in CLOSE_ONLY:
        df["prior_50_high"] = df["Close"].rolling(50).max().shift(1)
        df["prior_20_low"] = df["Close"].rolling(20).min().shift(1)
    else:
        df["prior_50_high"] = df["High"].rolling(50).max().shift(1)
        df["prior_20_low"] = df["Low"].rolling(20).min().shift(1)

    return df


def convert_price_for_cash_calcs(name: str, value: float) -> float:
    """
    Converts a displayed market price into the cash value used for:
    - position sizing
    - entry value
    - current value
    - P&L
    """
    scale = PRICE_SCALE.get(name, 1.0)
    return value * scale


def calculate_position_size(
    name: str,
    close: float,
    prior_20_low: float,
    status: str
) -> tuple[float | None, int | None, float | None]:
    if status != "BUY":
        return None, None, None

    close_for_sizing = convert_price_for_cash_calcs(name, close)
    stop_for_sizing = convert_price_for_cash_calcs(name, prior_20_low)

    risk_per_share = close_for_sizing - stop_for_sizing

    if risk_per_share <= 0:
        return None, None, None

    position_size = math.floor(RISK_PER_TRADE / risk_per_share)

    if position_size <= 0:
        return risk_per_share, 0, 0.0

    capital_required = position_size * close_for_sizing
    return risk_per_share, position_size, capital_required


def get_signal(name: str, symbol: str) -> SignalResult:
    df = download_ohlc(symbol)
    df = apply_symbol_fixes(name, symbol, df)
    df = compute_channels(name, df)

    df["daily_pct_change"] = df["Close"].pct_change() * 100

    last = df.iloc[-1]

    close = float(last["Close"])
    prior_50_high = float(last["prior_50_high"])
    prior_20_low = float(last["prior_20_low"])
    pct_change_from_yesterday = (
        float(last["daily_pct_change"]) if pd.notna(last["daily_pct_change"]) else 0.0
    )

    if math.isnan(prior_50_high) or math.isnan(prior_20_low):
        raise ValueError(f"Rolling values unavailable for {symbol}")

    entry_trigger = close > prior_50_high
    exit_trigger = close < prior_20_low

    pct_to_breakout = ((prior_50_high - close) / close) * 100
    pct_to_sale = ((close - prior_20_low) / prior_20_low) * 100

    if entry_trigger:
        status = "BUY"
    elif exit_trigger:
        status = "SELL"
    else:
        status = "HOLD"

    risk_per_share, position_size, capital_required = calculate_position_size(
        name=name,
        close=close,
        prior_20_low=prior_20_low,
        status=status,
    )

    return SignalResult(
        name=name,
        symbol=symbol,
        close=close,
        prior_50_high=prior_50_high,
        prior_20_low=prior_20_low,
        pct_to_breakout=pct_to_breakout,
        pct_to_sale=pct_to_sale,
        entry_trigger=entry_trigger,
        exit_trigger=exit_trigger,
        status=status,
        pct_change_from_yesterday=pct_change_from_yesterday,
        risk_per_share=risk_per_share,
        position_size=position_size,
        capital_required=capital_required,
    )


# -------------------------------
# Run full watchlist
# -------------------------------

def run_watchlist(watchlist: Dict[str, str]) -> tuple[pd.DataFrame, List[tuple[str, str, str]]]:
    results: List[SignalResult] = []
    errors: List[tuple[str, str, str]] = []

    for name, symbol in watchlist.items():
        try:
            results.append(get_signal(name, symbol))
        except Exception as exc:
            errors.append((name, symbol, str(exc)))

    if not results:
        return pd.DataFrame(), errors

    df = pd.DataFrame([r.__dict__ for r in results])

    df = df[
        [
            "name",
            "symbol",
            "close",
            "prior_50_high",
            "prior_20_low",
            "pct_to_breakout",
            "pct_to_sale",
            "pct_change_from_yesterday",
            "risk_per_share",
            "position_size",
            "capital_required",
            "status",
        ]
    ].sort_values(by=["status", "pct_to_breakout"], ascending=[True, True])

    return df, errors


# -------------------------------
# Portfolio
# -------------------------------

def load_portfolio() -> pd.DataFrame:
    try:
        df = pd.read_csv("portfolio.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    required = {"symbol", "name", "entry_price", "position_size", "entry_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"portfolio.csv is missing columns: {sorted(missing)}")

    df = df.copy()
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["position_size"] = pd.to_numeric(df["position_size"], errors="coerce")
    df["entry_date"] = df["entry_date"].astype(str)

    df = df.dropna(subset=["symbol", "name", "entry_price", "position_size"])

    return df


def calculate_portfolio_metrics(portfolio: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if portfolio.empty or signals.empty:
        return pd.DataFrame()

    current_prices = signals[["symbol", "name", "close", "status"]].copy()
    merged = portfolio.merge(current_prices, on=["symbol", "name"], how="left")

    if merged.empty:
        return pd.DataFrame()

    merged["entry_price_display"] = merged["entry_price"].astype(float)
    merged["current_price_display"] = pd.to_numeric(merged["close"], errors="coerce")
    merged["position_size"] = pd.to_numeric(merged["position_size"], errors="coerce")

    merged["entry_price_calc"] = merged.apply(
        lambda row: convert_price_for_cash_calcs(row["name"], float(row["entry_price_display"])),
        axis=1,
    )

    merged["current_price_calc"] = merged.apply(
        lambda row: convert_price_for_cash_calcs(row["name"], float(row["current_price_display"]))
        if pd.notna(row["current_price_display"]) else float("nan"),
        axis=1,
    )

    merged["pnl_per_share"] = merged["current_price_calc"] - merged["entry_price_calc"]
    merged["pnl_total"] = merged["pnl_per_share"] * merged["position_size"]
    merged["pnl_pct"] = (merged["pnl_per_share"] / merged["entry_price_calc"]) * 100

    merged["entry_value"] = merged["entry_price_calc"] * merged["position_size"]
    merged["current_value"] = merged["current_price_calc"] * merged["position_size"]

    merged["exit_signal"] = merged["status"].apply(lambda x: "SELL" if x == "SELL" else "")

    return merged[
        [
            "symbol",
            "name",
            "entry_date",
            "entry_price_display",
            "current_price_display",
            "position_size",
            "entry_value",
            "current_value",
            "pnl_total",
            "pnl_pct",
            "exit_signal",
        ]
    ]


# -------------------------------
# Formatting helpers
# -------------------------------

def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    formatted = df.copy()

    for col in ["close", "prior_50_high", "prior_20_low"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: f"{float(x):,.2f}" if pd.notna(x) else ""
            )

    for col in ["risk_per_share", "capital_required"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: f"{float(x):,.2f}" if pd.notna(x) else ""
            )

    for col in ["pct_to_breakout", "pct_to_sale", "pct_change_from_yesterday"]:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: f"{float(x):.2f}%" if pd.notna(x) else ""
            )

    if "position_size" in formatted.columns:
        formatted["position_size"] = formatted["position_size"].map(
            lambda x: f"{int(x):,}" if pd.notna(x) else ""
        )

    return formatted


def format_portfolio_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    formatted = df.copy()

    # Show actual quoted prices exactly as entered / downloaded
    for col in ["entry_price_display", "current_price_display"]:
        formatted[col] = formatted[col].map(
            lambda x: f"{float(x):,.2f}" if pd.notna(x) else ""
        )

    # Show cash/value columns in pounds
    for col in ["entry_value", "current_value", "pnl_total"]:
        formatted[col] = formatted[col].map(
            lambda x: f"£{float(x):,.2f}" if pd.notna(x) else ""
        )

    formatted["pnl_pct"] = formatted["pnl_pct"].map(
        lambda x: f"{float(x):.2f}%" if pd.notna(x) else ""
    )

    formatted["position_size"] = formatted["position_size"].map(
        lambda x: f"{int(x):,}" if pd.notna(x) else ""
    )

    return formatted

def build_summary_html(
    df: pd.DataFrame,
    errors: List[tuple[str, str, str]],
    portfolio_df: pd.DataFrame
) -> str:
    buy_count = 0
    sell_count = 0
    hold_count = 0

    if not df.empty:
        buy_count = int((df["status"] == "BUY").sum())
        sell_count = int((df["status"] == "SELL").sum())
        hold_count = int((df["status"] == "HOLD").sum())

    error_count = len(errors)

    total_current_value = 0.0
    total_pnl = 0.0
    open_positions = 0

    if not portfolio_df.empty:
        total_current_value = float(portfolio_df["current_value"].sum())
        total_pnl = float(portfolio_df["pnl_total"].sum())
        open_positions = len(portfolio_df)

    pnl_bg = "#d4edda" if total_pnl >= 0 else "#f8d7da"
    pnl_color = "#155724" if total_pnl >= 0 else "#721c24"

    return f"""
    <div style="margin-bottom: 20px;">
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #d4edda; color: #155724; border: 1px solid #c3e6cb; font-weight: bold;">
            BUY: {buy_count}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; font-weight: bold;">
            SELL: {sell_count}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #fff3cd; color: #856404; border: 1px solid #ffeeba; font-weight: bold;">
            HOLD: {hold_count}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; font-weight: bold;">
            ERRORS: {error_count}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #e9ecef; color: #212529; border: 1px solid #ced4da; font-weight: bold;">
            RISK / TRADE: £{RISK_PER_TRADE:,.2f}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #e9ecef; color: #212529; border: 1px solid #ced4da; font-weight: bold;">
            OPEN POSITIONS: {open_positions}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: #e9ecef; color: #212529; border: 1px solid #ced4da; font-weight: bold;">
            PORTFOLIO VALUE: £{total_current_value:,.2f}
        </div>
        <div style="display: inline-block; margin: 0 12px 12px 0; padding: 12px 16px; background: {pnl_bg}; color: {pnl_color}; border: 1px solid #ced4da; font-weight: bold;">
            TOTAL P&amp;L: £{total_pnl:,.2f}
        </div>
    </div>
    """


def build_actionable_html(df: pd.DataFrame) -> str:
    if df.empty:
        return """
        <p style="margin: 0 0 24px 0;">No signals returned.</p>
        """

    actionable = df[df["status"].isin(["BUY", "SELL"])].copy()

    if actionable.empty:
        return """
        <p style="margin: 0 0 24px 0;">No actionable BUY or SELL signals today.</p>
        """

    actionable = format_for_display(actionable)

    rows = []
    for _, row in actionable.iterrows():
        status = row["status"]

        if status == "BUY":
            status_bg = "#d4edda"
            status_color = "#155724"
        else:
            status_bg = "#f8d7da"
            status_color = "#721c24"

        rows.append(f"""
        <tr>
            <td style="padding:10px; border:1px solid #ddd;">{row['name']}</td>
            <td style="padding:10px; border:1px solid #ddd;">{row['symbol']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['close']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['risk_per_share']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['position_size']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['capital_required']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right; background:{status_bg}; color:{status_color}; font-weight:bold;">
                {status}
            </td>
        </tr>
        """)

    return f"""
    <h3 style="margin: 0 0 12px 0;">Actionable Signals</h3>
    <table style="border-collapse: collapse; width: 100%; font-size: 14px; margin-bottom: 24px;">
        <thead>
            <tr style="background: #060B69; color: #ffffff;">
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Name</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Symbol</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Close</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Risk / Share</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Position Size</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Capital Required</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Status</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def build_portfolio_html(portfolio_df: pd.DataFrame) -> str:
    if portfolio_df.empty:
        return """
        <h3 style="margin: 0 0 12px 0;">Portfolio</h3>
        <p style="margin: 0 0 24px 0;">No active positions in portfolio.csv.</p>
        """

    formatted = format_portfolio_for_display(portfolio_df)

    rows = []
    for idx, row in formatted.iterrows():
        raw_pnl = float(portfolio_df.loc[idx, "pnl_total"])
        pnl_bg = "#d4edda" if raw_pnl >= 0 else "#f8d7da"
        pnl_color = "#155724" if raw_pnl >= 0 else "#721c24"

        exit_signal = row["exit_signal"]
        exit_bg = "#f8d7da" if exit_signal == "SELL" else "#ffffff"
        exit_color = "#721c24" if exit_signal == "SELL" else "#222222"

        rows.append(f"""
        <tr>
            <td style="padding:10px; border:1px solid #ddd;">{row['name']}</td>
            <td style="padding:10px; border:1px solid #ddd;">{row['symbol']}</td>
            <td style="padding:10px; border:1px solid #ddd;">{row['entry_date']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['entry_price_display']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['current_price_display']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['position_size']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['entry_value']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['current_value']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right; background:{pnl_bg}; color:{pnl_color}; font-weight:bold;">{row['pnl_total']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right; background:{pnl_bg}; color:{pnl_color}; font-weight:bold;">{row['pnl_pct']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:center; background:{exit_bg}; color:{exit_color}; font-weight:bold;">{exit_signal}</td>
        </tr>
        """)

    return f"""
    <h3 style="margin: 0 0 12px 0;">Portfolio</h3>
    <table style="border-collapse: collapse; width: 100%; font-size: 14px; margin-bottom: 24px;">
        <thead>
            <tr style="background: #060B69; color: #ffffff;">
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Name</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Symbol</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Entry Date</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Entry Price</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Current Price</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Size</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Entry Value</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Current Value</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">P&amp;L</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">P&amp;L %</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:center;">Exit Signal</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def build_full_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return """
        <h3 style="margin: 0 0 12px 0;">Full Watchlist</h3>
        <p>No data returned.</p>
        """

    formatted = format_for_display(df)

    rows = []
    for _, row in formatted.iterrows():
        status = row["status"]

        if status == "BUY":
            status_bg = "#d4edda"
            status_color = "#155724"
        elif status == "SELL":
            status_bg = "#f8d7da"
            status_color = "#721c24"
        else:
            status_bg = "#fff3cd"
            status_color = "#856404"

        rows.append(f"""
        <tr>
            <td style="padding:10px; border:1px solid #ddd;">{row['name']}</td>
            <td style="padding:10px; border:1px solid #ddd;">{row['symbol']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['close']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['prior_50_high']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['prior_20_low']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['pct_to_breakout']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['pct_to_sale']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['pct_change_from_yesterday']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['risk_per_share']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['position_size']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right;">{row['capital_required']}</td>
            <td style="padding:10px; border:1px solid #ddd; text-align:right; background:{status_bg}; color:{status_color}; font-weight:bold;">
                {status}
            </td>
        </tr>
        """)

    return f"""
    <h3 style="margin: 0 0 12px 0;">Full Watchlist</h3>
    <table style="border-collapse: collapse; width: 100%; font-size: 14px;">
        <thead>
            <tr style="background: #060B69; color: #ffffff;">
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Name</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:left;">Symbol</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Close</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Prior 50 High</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Prior 20 Low</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">% to Breakout</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">% to Sale</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">% Change</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Risk / Share</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Position Size</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Capital Required</th>
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Status</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def build_errors_html(errors: List[tuple[str, str, str]]) -> str:
    if not errors:
        return ""

    items = []
    for name, symbol, msg in errors:
        items.append(f"<li><strong>{name}</strong> ({symbol}): {msg}</li>")

    return f"""
    <h3 style="margin: 24px 0 12px 0;">Errors</h3>
    <ul style="margin: 0; padding-left: 20px;">
        {''.join(items)}
    </ul>
    """


def build_html_email(
    df: pd.DataFrame,
    errors: List[tuple[str, str, str]],
    portfolio_df: pd.DataFrame
) -> str:
    summary_html = build_summary_html(df, errors, portfolio_df)
    actionable_html = build_actionable_html(df)
    portfolio_html = build_portfolio_html(portfolio_df)
    full_table_html = build_full_table_html(df)
    errors_html = build_errors_html(errors)

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #222; margin: 0; padding: 24px; background: #f7f7f7;">
        <div style="max-width: 1400px; margin: 0 auto; background: #ffffff; padding: 24px; border: 1px solid #e5e5e5;">
            <h2 style="margin-top: 0;">Daily Market Check</h2>
            <p style="margin: 0 0 18px 0;">Donchian 50 / 20 watchlist scan with position sizing and portfolio tracking.</p>

            {summary_html}
            {actionable_html}
            {portfolio_html}
            {full_table_html}
            {errors_html}
        </div>
    </body>
    </html>
    """


# -------------------------------
# Main execution
# -------------------------------

def main() -> None:
    signals_df, errors = run_watchlist(WATCHLIST)
    portfolio_input_df = load_portfolio()
    portfolio_metrics_df = calculate_portfolio_metrics(portfolio_input_df, signals_df)

    print("Daily Market Check")
    print("=" * 100)

    if signals_df.empty:
        print("No signals returned.")
    else:
        print("\nSignals")
        print(format_for_display(signals_df).to_string(index=False))

    if not portfolio_metrics_df.empty:
        print("\nPortfolio")
        print(format_portfolio_for_display(portfolio_metrics_df).to_string(index=False))
    else:
        print("\nPortfolio")
        print("No active positions in portfolio.csv.")

    if errors:
        print("\nErrors:")
        for name, symbol, msg in errors:
            print(f"- {name} ({symbol}): {msg}")

    html = build_html_email(signals_df, errors, portfolio_metrics_df)

    with open("market_email.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()