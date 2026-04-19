from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import math

import pandas as pd
import yfinance as yf


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
    # Sometimes only the latest close is shown in pence rather than pounds
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
            "status",
        ]
    ].sort_values(by=["status", "pct_to_breakout"], ascending=[True, True])

    return df, errors


# -------------------------------
# Formatting helpers
# -------------------------------

def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    formatted = df.copy()

    for col in ["close", "prior_50_high", "prior_20_low"]:
        formatted[col] = formatted[col].map(lambda x: f"{float(x):,.2f}")

    for col in ["pct_to_breakout", "pct_to_sale", "pct_change_from_yesterday"]:
        formatted[col] = formatted[col].map(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "")

    return formatted


def build_summary_html(df: pd.DataFrame, errors: List[tuple[str, str, str]]) -> str:
    buy_count = 0
    sell_count = 0
    hold_count = 0

    if not df.empty:
        buy_count = int((df["status"] == "BUY").sum())
        sell_count = int((df["status"] == "SELL").sum())
        hold_count = int((df["status"] == "HOLD").sum())

    error_count = len(errors)

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
                <th style="padding:10px; border:1px solid #ddd; text-align:right;">Status</th>
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
        items.append(
            f"<li><strong>{name}</strong> ({symbol}): {msg}</li>"
        )

    return f"""
    <h3 style="margin: 24px 0 12px 0;">Errors</h3>
    <ul style="margin: 0; padding-left: 20px;">
        {''.join(items)}
    </ul>
    """


def build_html_email(df: pd.DataFrame, errors: List[tuple[str, str, str]]) -> str:
    summary_html = build_summary_html(df, errors)
    actionable_html = build_actionable_html(df)
    full_table_html = build_full_table_html(df)
    errors_html = build_errors_html(errors)

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #222; margin: 0; padding: 24px; background: #f7f7f7;">
        <div style="max-width: 1100px; margin: 0 auto; background: #ffffff; padding: 24px; border: 1px solid #e5e5e5;">
            <h2 style="margin-top: 0;">Daily Market Check</h2>
            <p style="margin: 0 0 18px 0;">Donchian 50 / 20 watchlist scan.</p>

            {summary_html}
            {actionable_html}
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

    print("Daily Market Check")
    print("=" * 80)

    if signals_df.empty:
        print("No signals returned.")
    else:
        print(format_for_display(signals_df).to_string(index=False))

    if errors:
        print("\nErrors:")
        for name, symbol, msg in errors:
            print(f"- {name} ({symbol}): {msg}")

    html = build_html_email(signals_df, errors)

    with open("market_email.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
