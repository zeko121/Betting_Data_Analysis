"""Wallet data loading and crypto price utilities."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import time
import streamlit as st

# Stablecoins that are 1:1 with USD (no conversion needed)
STABLECOINS = {
    'usdt', 'usdc', 'busd', 'dai', 'tusd', 'usdp', 'gusd', 'frax',
    'lusd', 'susd', 'usdd', 'ust', 'mim', 'fei', 'usd'
}

# Binance symbol mapping for common cryptocurrencies
BINANCE_SYMBOLS = {
    'btc': 'BTCUSDT',
    'bitcoin': 'BTCUSDT',
    'eth': 'ETHUSDT',
    'ethereum': 'ETHUSDT',
    'doge': 'DOGEUSDT',
    'dogecoin': 'DOGEUSDT',
    'ltc': 'LTCUSDT',
    'litecoin': 'LTCUSDT',
    'xrp': 'XRPUSDT',
    'ripple': 'XRPUSDT',
    'ada': 'ADAUSDT',
    'cardano': 'ADAUSDT',
    'sol': 'SOLUSDT',
    'solana': 'SOLUSDT',
    'dot': 'DOTUSDT',
    'polkadot': 'DOTUSDT',
    'matic': 'MATICUSDT',
    'polygon': 'MATICUSDT',
    'avax': 'AVAXUSDT',
    'avalanche': 'AVAXUSDT',
    'link': 'LINKUSDT',
    'chainlink': 'LINKUSDT',
    'atom': 'ATOMUSDT',
    'cosmos': 'ATOMUSDT',
    'uni': 'UNIUSDT',
    'uniswap': 'UNIUSDT',
    'shib': 'SHIBUSDT',
    'bnb': 'BNBUSDT',
    'trx': 'TRXUSDT',
    'tron': 'TRXUSDT',
    'xlm': 'XLMUSDT',
    'stellar': 'XLMUSDT',
    'near': 'NEARUSDT',
    'algo': 'ALGOUSDT',
    'ftm': 'FTMUSDT',
    'sand': 'SANDUSDT',
    'mana': 'MANAUSDT',
    'axs': 'AXSUSDT',
    'ape': 'APEUSDT',
    'crv': 'CRVUSDT',
    'aave': 'AAVEUSDT',
    'mkr': 'MKRUSDT',
    'snx': 'SNXUSDT',
    'sushi': 'SUSHIUSDT',
    '1inch': '1INCHUSDT',
    'ens': 'ENSUSDT',
    'grt': 'GRTUSDT',
    'ldo': 'LDOUSDT',
    'op': 'OPUSDT',
    'arb': 'ARBUSDT',
    'pepe': 'PEPEUSDT',
    'bonk': 'BONKUSDT',
    'wif': 'WIFUSDT',
    'etc': 'ETCUSDT',
    'bch': 'BCHUSDT',
    'fil': 'FILUSDT',
    'apt': 'APTUSDT',
    'inj': 'INJUSDT',
    'sei': 'SEIUSDT',
}

# Fallback average prices by year (rough estimates for when API fails)
FALLBACK_PRICES = {
    'BTCUSDT': {2021: 47000, 2022: 28000, 2023: 30000, 2024: 45000, 2025: 95000},
    'ETHUSDT': {2021: 3000, 2022: 1800, 2023: 1800, 2024: 2500, 2025: 3500},
    'DOGEUSDT': {2021: 0.15, 2022: 0.08, 2023: 0.07, 2024: 0.12, 2025: 0.35},
    'LTCUSDT': {2021: 150, 2022: 70, 2023: 80, 2024: 75, 2025: 120},
    'XRPUSDT': {2021: 0.80, 2022: 0.40, 2023: 0.50, 2024: 0.55, 2025: 2.20},
    'ADAUSDT': {2021: 1.50, 2022: 0.40, 2023: 0.35, 2024: 0.45, 2025: 1.00},
    'SOLUSDT': {2021: 100, 2022: 30, 2023: 25, 2024: 120, 2025: 220},
    'BNBUSDT': {2021: 400, 2022: 300, 2023: 250, 2024: 350, 2025: 700},
    'MATICUSDT': {2021: 1.50, 2022: 0.90, 2023: 0.80, 2024: 0.70, 2025: 0.50},
    'DOTUSDT': {2021: 30, 2022: 8, 2023: 6, 2024: 7, 2025: 8},
    'AVAXUSDT': {2021: 80, 2022: 20, 2023: 15, 2024: 35, 2025: 45},
    'LINKUSDT': {2021: 25, 2022: 8, 2023: 10, 2024: 15, 2025: 25},
    'TRXUSDT': {2021: 0.08, 2022: 0.06, 2023: 0.08, 2024: 0.12, 2025: 0.25},
    'XLMUSDT': {2021: 0.30, 2022: 0.12, 2023: 0.12, 2024: 0.12, 2025: 0.45},
    'SHIBUSDT': {2021: 0.00003, 2022: 0.00001, 2023: 0.000009, 2024: 0.00002, 2025: 0.00002},
}


def parse_wallet_date(date_str: str) -> datetime:
    """
    Parse the date format from wallet CSV files.
    Format: "Wed Dec 10 2025 19:58:43 GMT+0000 (Coordinated Universal Time)"
    """
    try:
        # Remove the timezone name in parentheses
        if '(' in date_str:
            date_str = date_str.split('(')[0].strip()

        # Parse the date
        return datetime.strptime(date_str, "%a %b %d %Y %H:%M:%S GMT%z")
    except ValueError:
        try:
            # Try alternative format without timezone
            date_str_clean = date_str.split('GMT')[0].strip()
            return datetime.strptime(date_str_clean, "%a %b %d %Y %H:%M:%S")
        except ValueError:
            # Return current time if parsing fails
            return datetime.now()


def load_wallet_csv(file_content: bytes) -> pd.DataFrame:
    """
    Load a deposit or withdrawal CSV file.

    Args:
        file_content: Raw bytes from uploaded file

    Returns:
        DataFrame with parsed wallet transactions
    """
    import io

    # Read CSV
    df = pd.read_csv(io.BytesIO(file_content))

    # Ensure required columns exist
    required_cols = ['date', 'amount', 'currency']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Parse dates
    df['timestamp'] = df['date'].apply(parse_wallet_date)
    df['date_only'] = df['timestamp'].apply(lambda x: x.date() if hasattr(x, 'date') else x)

    # Normalize currency names
    df['currency'] = df['currency'].str.lower().str.strip()

    # Convert amount to float
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

    # Identify if stablecoin
    df['is_stablecoin'] = df['currency'].isin(STABLECOINS)

    # Sort by date
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def get_binance_symbol(currency: str) -> Optional[str]:
    """Get Binance trading pair symbol for a currency."""
    currency_lower = currency.lower().strip()
    return BINANCE_SYMBOLS.get(currency_lower)


def get_historical_price_binance(symbol: str, date: datetime, status_callback=None) -> Optional[float]:
    """
    Get historical price for a cryptocurrency on a specific date using Binance API.

    Args:
        symbol: Binance trading pair symbol (e.g., 'BTCUSDT')
        date: datetime object for the date
        status_callback: Optional callback for status updates

    Returns:
        Price in USD or None if not found
    """
    try:
        # Convert date to milliseconds timestamp (start of day UTC)
        start_time = int(datetime(date.year, date.month, date.day).timestamp() * 1000)
        end_time = start_time + (24 * 60 * 60 * 1000)  # End of day

        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': '1d',  # Daily candle
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1
        }

        if status_callback:
            status_callback(f"    🌐 Binance API: {symbol} for {date.strftime('%Y-%m-%d')}")

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                # Kline format: [open_time, open, high, low, close, volume, ...]
                # Use the close price (index 4)
                close_price = float(data[0][4])
                if status_callback:
                    status_callback(f"    ✅ Got price: ${close_price:,.6f}")
                return close_price
            else:
                if status_callback:
                    status_callback(f"    ⚠️ No data returned for this date")
        elif response.status_code == 429:
            if status_callback:
                status_callback(f"    ⏳ Rate limited, waiting 2 seconds...")
            time.sleep(2)
            return get_historical_price_binance(symbol, date, None)
        else:
            if status_callback:
                status_callback(f"    ❌ API error: {response.status_code}")

        return None
    except requests.exceptions.Timeout:
        if status_callback:
            status_callback(f"    ⏱️ Request timed out")
        return None
    except Exception as e:
        if status_callback:
            status_callback(f"    ❌ Exception: {str(e)}")
        return None


def get_fallback_price(symbol: str, year: int) -> Optional[float]:
    """Get fallback price estimate for a symbol in a given year."""
    if symbol in FALLBACK_PRICES:
        year_prices = FALLBACK_PRICES[symbol]
        if year in year_prices:
            return year_prices[year]
        # Find closest year
        years = sorted(year_prices.keys())
        if year < years[0]:
            return year_prices[years[0]]
        if year > years[-1]:
            return year_prices[years[-1]]
    return None


def calculate_usd_values(df: pd.DataFrame, progress_callback=None, status_callback=None) -> pd.DataFrame:
    """
    Calculate USD values for all transactions.

    Args:
        df: DataFrame with wallet transactions
        progress_callback: Optional callback for progress bar updates (0-1)
        status_callback: Optional callback for status text updates

    Returns:
        DataFrame with USD values added
    """
    df = df.copy()
    df['usd_value'] = 0.0
    df['price_at_time'] = 0.0
    df['price_source'] = ''

    # Group by currency for efficiency
    currencies = df['currency'].unique()
    total_txs = len(df)
    processed = 0

    if status_callback:
        status_callback(f"📊 Processing {total_txs} transactions across {len(currencies)} currencies...")

    for currency in currencies:
        mask = df['currency'] == currency
        currency_txs = df[mask]
        num_txs = len(currency_txs)

        if currency in STABLECOINS:
            # Stablecoins are 1:1 with USD
            if status_callback:
                status_callback(f"💵 {currency.upper()}: {num_txs} transactions (stablecoin, 1:1 USD)")
            df.loc[mask, 'usd_value'] = df.loc[mask, 'amount']
            df.loc[mask, 'price_at_time'] = 1.0
            df.loc[mask, 'price_source'] = 'stablecoin (1:1)'
            processed += num_txs
            if progress_callback:
                progress_callback(processed / total_txs)
            continue

        # Get Binance symbol
        symbol = get_binance_symbol(currency)

        if not symbol:
            # Unknown currency - mark for manual review
            if status_callback:
                status_callback(f"⚠️ {currency.upper()}: {num_txs} transactions (unknown currency, skipping)")
            df.loc[mask, 'price_source'] = 'unknown currency'
            processed += num_txs
            if progress_callback:
                progress_callback(processed / total_txs)
            continue

        # Fetch historical prices from Binance API
        unique_dates = currency_txs['date_only'].unique()
        date_prices = {}

        if status_callback:
            status_callback(f"🔄 {currency.upper()}: Fetching prices for {len(unique_dates)} unique dates ({num_txs} transactions)...")

        for i, date in enumerate(unique_dates):
            if status_callback:
                status_callback(f"🌐 {currency.upper()}: Fetching price for {date} ({i+1}/{len(unique_dates)})...")

            # Try Binance API first
            price = get_historical_price_binance(symbol, date, status_callback)

            if price is not None:
                date_prices[date] = price
            else:
                # Fall back to yearly estimate
                fallback = get_fallback_price(symbol, date.year)
                if fallback:
                    date_prices[date] = fallback
                    if status_callback:
                        status_callback(f"    📊 Using fallback estimate: ${fallback:,.4f}")

            # Small delay to avoid rate limiting
            time.sleep(0.2)

        # Apply prices to transactions
        if status_callback:
            status_callback(f"📝 {currency.upper()}: Applying prices to {num_txs} transactions...")

        for idx in currency_txs.index:
            tx_date = df.loc[idx, 'date_only']
            if tx_date in date_prices:
                price = date_prices[tx_date]
                df.loc[idx, 'price_at_time'] = price
                df.loc[idx, 'usd_value'] = df.loc[idx, 'amount'] * price
                df.loc[idx, 'price_source'] = f'binance ({symbol})'
            else:
                df.loc[idx, 'price_source'] = 'no price available'

            processed += 1
            if progress_callback:
                progress_callback(processed / total_txs)

    if status_callback:
        status_callback(f"✅ Done! Processed {processed} transactions.")

    return df


def calculate_wallet_summary(
    deposits_df: pd.DataFrame,
    withdrawals_df: pd.DataFrame
) -> Dict:
    """
    Calculate wallet summary statistics.

    Args:
        deposits_df: DataFrame with deposit transactions (with USD values)
        withdrawals_df: DataFrame with withdrawal transactions (with USD values)

    Returns:
        Dictionary with summary statistics
    """
    summary = {}

    # Handle empty dataframes
    has_deposits = len(deposits_df) > 0 and 'usd_value' in deposits_df.columns
    has_withdrawals = len(withdrawals_df) > 0 and 'usd_value' in withdrawals_df.columns

    # Total deposits
    summary['total_deposits_usd'] = float(deposits_df['usd_value'].sum()) if has_deposits else 0.0
    summary['total_deposit_count'] = len(deposits_df) if has_deposits else 0

    # Total withdrawals
    summary['total_withdrawals_usd'] = float(withdrawals_df['usd_value'].sum()) if has_withdrawals else 0.0
    summary['total_withdrawal_count'] = len(withdrawals_df) if has_withdrawals else 0

    # Net profit (withdrawals - deposits)
    summary['net_profit_usd'] = summary['total_withdrawals_usd'] - summary['total_deposits_usd']
    summary['roi_percentage'] = (
        (summary['net_profit_usd'] / summary['total_deposits_usd'] * 100)
        if summary['total_deposits_usd'] > 0 else 0.0
    )

    # Breakdown by currency (deposits) - store as simple values, not nested dicts
    if has_deposits:
        deposits_by_currency = deposits_df.groupby('currency').agg({
            'amount': 'sum',
            'usd_value': 'sum',
        })
        deposits_by_currency['count'] = deposits_df.groupby('currency').size()
        summary['deposit_currencies'] = list(deposits_by_currency.index)
        summary['deposit_amounts_by_currency'] = deposits_by_currency['amount'].to_dict()
        summary['deposit_usd_by_currency'] = deposits_by_currency['usd_value'].to_dict()
    else:
        summary['deposit_currencies'] = []
        summary['deposit_amounts_by_currency'] = {}
        summary['deposit_usd_by_currency'] = {}

    # Breakdown by currency (withdrawals) - store as simple values, not nested dicts
    if has_withdrawals:
        withdrawals_by_currency = withdrawals_df.groupby('currency').agg({
            'amount': 'sum',
            'usd_value': 'sum',
        })
        withdrawals_by_currency['count'] = withdrawals_df.groupby('currency').size()
        summary['withdrawal_currencies'] = list(withdrawals_by_currency.index)
        summary['withdrawal_amounts_by_currency'] = withdrawals_by_currency['amount'].to_dict()
        summary['withdrawal_usd_by_currency'] = withdrawals_by_currency['usd_value'].to_dict()
    else:
        summary['withdrawal_currencies'] = []
        summary['withdrawal_amounts_by_currency'] = {}
        summary['withdrawal_usd_by_currency'] = {}

    # Date range
    date_parts = []
    if has_deposits and 'timestamp' in deposits_df.columns:
        date_parts.append(deposits_df['timestamp'])
    if has_withdrawals and 'timestamp' in withdrawals_df.columns:
        date_parts.append(withdrawals_df['timestamp'])

    if date_parts:
        all_dates = pd.concat(date_parts)
        if len(all_dates) > 0:
            summary['first_transaction'] = all_dates.min()
            summary['last_transaction'] = all_dates.max()

    # Transactions with missing prices
    deposits_missing = len(deposits_df[deposits_df['usd_value'] == 0]) if has_deposits else 0
    withdrawals_missing = len(withdrawals_df[withdrawals_df['usd_value'] == 0]) if has_withdrawals else 0
    summary['missing_prices'] = deposits_missing + withdrawals_missing

    return summary


def create_wallet_charts(
    deposits_df: pd.DataFrame,
    withdrawals_df: pd.DataFrame,
    summary: Dict
) -> Dict:
    """Create charts for wallet analysis."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from charts.theme import COLORS, apply_dark_theme

    charts = {}

    # Check for valid data
    has_deposits = (deposits_df is not None and len(deposits_df) > 0
                    and 'timestamp' in deposits_df.columns
                    and 'usd_value' in deposits_df.columns)
    has_withdrawals = (withdrawals_df is not None and len(withdrawals_df) > 0
                       and 'timestamp' in withdrawals_df.columns
                       and 'usd_value' in withdrawals_df.columns)

    # -------------------------------------------------------------------------
    # 1. Deposits vs Withdrawals Over Time
    # -------------------------------------------------------------------------
    fig_timeline = go.Figure()

    # Group by month for cleaner visualization
    if has_deposits:
        deposits_monthly = deposits_df.groupby(
            deposits_df['timestamp'].dt.to_period('M')
        )['usd_value'].sum()
        deposits_monthly.index = deposits_monthly.index.to_timestamp()
    else:
        deposits_monthly = pd.Series(dtype=float)

    if has_withdrawals:
        withdrawals_monthly = withdrawals_df.groupby(
            withdrawals_df['timestamp'].dt.to_period('M')
        )['usd_value'].sum()
        withdrawals_monthly.index = withdrawals_monthly.index.to_timestamp()
    else:
        withdrawals_monthly = pd.Series(dtype=float)

    fig_timeline.add_trace(go.Bar(
        x=deposits_monthly.index,
        y=deposits_monthly.values,
        name='Deposits',
        marker_color=COLORS['loss'],  # Red for money going in
        hovertemplate='Deposits: $%{y:,.2f}<extra></extra>',
    ))

    fig_timeline.add_trace(go.Bar(
        x=withdrawals_monthly.index,
        y=withdrawals_monthly.values,
        name='Withdrawals',
        marker_color=COLORS['profit'],  # Green for money coming out
        hovertemplate='Withdrawals: $%{y:,.2f}<extra></extra>',
    ))

    apply_dark_theme(
        fig_timeline,
        title='Deposits vs Withdrawals Over Time',
        xaxis_title='Month',
        yaxis_title='Amount (USD)',
        barmode='group',
    )

    charts['timeline'] = fig_timeline

    # -------------------------------------------------------------------------
    # 2. Currency Breakdown Pie Charts
    # -------------------------------------------------------------------------
    fig_breakdown = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'pie'}]],
        subplot_titles=('Deposits by Currency', 'Withdrawals by Currency')
    )

    # Deposits pie
    if has_deposits and 'currency' in deposits_df.columns:
        dep_by_curr = deposits_df.groupby('currency')['usd_value'].sum()
        fig_breakdown.add_trace(go.Pie(
            labels=dep_by_curr.index.tolist(),
            values=dep_by_curr.values.tolist(),
            hole=0.4,
            textinfo='label+percent',
            hovertemplate='%{label}: $%{value:,.2f}<extra></extra>',
        ), row=1, col=1)
    else:
        fig_breakdown.add_trace(go.Pie(
            labels=['No Data'],
            values=[1],
            hole=0.4,
            textinfo='label',
        ), row=1, col=1)

    # Withdrawals pie
    if has_withdrawals and 'currency' in withdrawals_df.columns:
        wit_by_curr = withdrawals_df.groupby('currency')['usd_value'].sum()
        fig_breakdown.add_trace(go.Pie(
            labels=wit_by_curr.index.tolist(),
            values=wit_by_curr.values.tolist(),
            hole=0.4,
            textinfo='label+percent',
            hovertemplate='%{label}: $%{value:,.2f}<extra></extra>',
        ), row=1, col=2)
    else:
        fig_breakdown.add_trace(go.Pie(
            labels=['No Data'],
            values=[1],
            hole=0.4,
            textinfo='label',
        ), row=1, col=2)

    apply_dark_theme(
        fig_breakdown,
        title='Currency Breakdown',
    )

    charts['breakdown'] = fig_breakdown

    # -------------------------------------------------------------------------
    # 3. Cumulative Flow Chart
    # -------------------------------------------------------------------------
    # Combine all transactions
    tx_parts = []
    if has_deposits:
        tx_parts.append(deposits_df[['timestamp', 'usd_value']].assign(type='deposit'))
    if has_withdrawals:
        tx_parts.append(withdrawals_df[['timestamp', 'usd_value']].assign(type='withdrawal'))

    if tx_parts:
        all_txs = pd.concat(tx_parts).sort_values('timestamp')
    else:
        all_txs = pd.DataFrame(columns=['timestamp', 'usd_value', 'type'])

    if len(all_txs) > 0:
        all_txs['net_value'] = all_txs.apply(
            lambda x: -x['usd_value'] if x['type'] == 'deposit' else x['usd_value'],
            axis=1
        )
        all_txs['cumulative'] = all_txs['net_value'].cumsum()

        fig_cumulative = go.Figure()

        fig_cumulative.add_trace(go.Scatter(
            x=all_txs['timestamp'],
            y=all_txs['cumulative'],
            mode='lines',
            name='Cumulative Balance',
            line=dict(color=COLORS['neutral'], width=2),
            fill='tozeroy',
            fillcolor='rgba(93, 173, 226, 0.2)',
            hovertemplate='%{x}<br>Balance: $%{y:,.2f}<extra></extra>',
        ))

        fig_cumulative.add_hline(y=0, line_dash='dash', line_color=COLORS['muted'])

        apply_dark_theme(
            fig_cumulative,
            title='Cumulative Balance (Withdrawals - Deposits)',
            xaxis_title='Date',
            yaxis_title='Balance (USD)',
        )

        charts['cumulative'] = fig_cumulative

    # -------------------------------------------------------------------------
    # 4. Summary Gauge Chart
    # -------------------------------------------------------------------------
    net_profit = summary.get('net_profit_usd', 0)
    total_deposits = summary.get('total_deposits_usd', 1)
    roi = summary.get('roi_percentage', 0)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=roi,
        title={'text': "Return on Investment (ROI)", 'font': {'color': COLORS['text']}},
        delta={'reference': 0, 'increasing': {'color': COLORS['profit']}, 'decreasing': {'color': COLORS['loss']}},
        number={'suffix': '%', 'font': {'color': COLORS['text']}},
        gauge={
            'axis': {'range': [-100, 100], 'tickcolor': COLORS['text']},
            'bar': {'color': COLORS['profit'] if roi >= 0 else COLORS['loss']},
            'bgcolor': COLORS['background'],
            'borderwidth': 2,
            'bordercolor': COLORS['grid'],
            'steps': [
                {'range': [-100, 0], 'color': 'rgba(255, 71, 87, 0.3)'},
                {'range': [0, 100], 'color': 'rgba(0, 212, 170, 0.3)'}
            ],
            'threshold': {
                'line': {'color': COLORS['text'], 'width': 4},
                'thickness': 0.75,
                'value': roi
            }
        }
    ))

    apply_dark_theme(fig_gauge, title='')
    charts['gauge'] = fig_gauge

    return charts
