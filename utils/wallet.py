"""Wallet data loading and crypto price utilities."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import requests
import time
import streamlit as st

# Stablecoins that are 1:1 with USD (no conversion needed)
STABLECOINS = {
    'usdt', 'usdc', 'busd', 'dai', 'tusd', 'usdp', 'gusd', 'frax',
    'lusd', 'susd', 'usdd', 'ust', 'mim', 'fei', 'usd'
}

# CoinGecko ID mapping for common cryptocurrencies
COINGECKO_IDS = {
    'btc': 'bitcoin',
    'bitcoin': 'bitcoin',
    'eth': 'ethereum',
    'ethereum': 'ethereum',
    'doge': 'dogecoin',
    'dogecoin': 'dogecoin',
    'ltc': 'litecoin',
    'litecoin': 'litecoin',
    'xrp': 'ripple',
    'ripple': 'ripple',
    'ada': 'cardano',
    'cardano': 'cardano',
    'sol': 'solana',
    'solana': 'solana',
    'dot': 'polkadot',
    'polkadot': 'polkadot',
    'matic': 'matic-network',
    'polygon': 'matic-network',
    'avax': 'avalanche-2',
    'avalanche': 'avalanche-2',
    'link': 'chainlink',
    'chainlink': 'chainlink',
    'atom': 'cosmos',
    'cosmos': 'cosmos',
    'uni': 'uniswap',
    'uniswap': 'uniswap',
    'shib': 'shiba-inu',
    'bnb': 'binancecoin',
    'trx': 'tron',
    'tron': 'tron',
    'xlm': 'stellar',
    'stellar': 'stellar',
    'near': 'near',
    'algo': 'algorand',
    'ftm': 'fantom',
    'sand': 'the-sandbox',
    'mana': 'decentraland',
    'axs': 'axie-infinity',
    'ape': 'apecoin',
    'crv': 'curve-dao-token',
    'aave': 'aave',
    'mkr': 'maker',
    'comp': 'compound-governance-token',
    'snx': 'havven',
    'sushi': 'sushi',
    '1inch': '1inch',
    'ens': 'ethereum-name-service',
    'grt': 'the-graph',
    'ldo': 'lido-dao',
    'rpl': 'rocket-pool',
    'op': 'optimism',
    'arb': 'arbitrum',
    'pepe': 'pepe',
    'bonk': 'bonk',
    'wif': 'dogwifcoin',
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


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_historical_price(coin_id: str, date: str) -> Optional[float]:
    """
    Get historical price for a cryptocurrency on a specific date.
    Uses CoinGecko API.

    Args:
        coin_id: CoinGecko coin ID
        date: Date in DD-MM-YYYY format

    Returns:
        Price in USD or None if not found
    """
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/history"
        params = {
            'date': date,
            'localization': 'false'
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if 'market_data' in data and 'current_price' in data['market_data']:
                return data['market_data']['current_price'].get('usd')
        elif response.status_code == 429:
            # Rate limited - wait and retry
            time.sleep(2)
            return get_historical_price(coin_id, date)

        return None
    except Exception as e:
        st.warning(f"Error fetching price for {coin_id} on {date}: {e}")
        return None


def get_coingecko_id(currency: str) -> Optional[str]:
    """Get CoinGecko ID for a currency symbol."""
    currency_lower = currency.lower().strip()
    return COINGECKO_IDS.get(currency_lower)


def calculate_usd_values(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    Calculate USD values for all transactions.

    Args:
        df: DataFrame with wallet transactions
        progress_callback: Optional callback for progress updates

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

    for currency in currencies:
        mask = df['currency'] == currency
        currency_txs = df[mask]

        if currency in STABLECOINS:
            # Stablecoins are 1:1 with USD
            df.loc[mask, 'usd_value'] = df.loc[mask, 'amount']
            df.loc[mask, 'price_at_time'] = 1.0
            df.loc[mask, 'price_source'] = 'stablecoin (1:1)'
            processed += len(currency_txs)
            if progress_callback:
                progress_callback(processed / total_txs)
            continue

        # Get CoinGecko ID
        coin_id = get_coingecko_id(currency)

        if not coin_id:
            # Unknown currency - mark for manual review
            df.loc[mask, 'price_source'] = 'unknown currency'
            processed += len(currency_txs)
            if progress_callback:
                progress_callback(processed / total_txs)
            continue

        # Fetch historical prices for each unique date
        unique_dates = currency_txs['date_only'].unique()
        date_prices = {}

        for date in unique_dates:
            # Format date for CoinGecko API
            date_str = date.strftime('%d-%m-%Y')
            price = get_historical_price(coin_id, date_str)

            if price is not None:
                date_prices[date] = price

            # Small delay to avoid rate limiting
            time.sleep(0.5)

        # Apply prices to transactions
        for idx in currency_txs.index:
            tx_date = df.loc[idx, 'date_only']
            if tx_date in date_prices:
                price = date_prices[tx_date]
                df.loc[idx, 'price_at_time'] = price
                df.loc[idx, 'usd_value'] = df.loc[idx, 'amount'] * price
                df.loc[idx, 'price_source'] = f'coingecko ({coin_id})'
            else:
                df.loc[idx, 'price_source'] = 'price not found'

            processed += 1
            if progress_callback:
                progress_callback(processed / total_txs)

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

    # Total deposits
    summary['total_deposits_usd'] = deposits_df['usd_value'].sum()
    summary['total_deposit_count'] = len(deposits_df)

    # Total withdrawals
    summary['total_withdrawals_usd'] = withdrawals_df['usd_value'].sum()
    summary['total_withdrawal_count'] = len(withdrawals_df)

    # Net profit (withdrawals - deposits)
    summary['net_profit_usd'] = summary['total_withdrawals_usd'] - summary['total_deposits_usd']
    summary['roi_percentage'] = (
        (summary['net_profit_usd'] / summary['total_deposits_usd'] * 100)
        if summary['total_deposits_usd'] > 0 else 0
    )

    # Breakdown by currency (deposits)
    deposits_by_currency = deposits_df.groupby('currency').agg({
        'amount': 'sum',
        'usd_value': 'sum',
        'currency': 'count'
    }).rename(columns={'currency': 'count'})
    summary['deposits_by_currency'] = deposits_by_currency.to_dict('index')

    # Breakdown by currency (withdrawals)
    withdrawals_by_currency = withdrawals_df.groupby('currency').agg({
        'amount': 'sum',
        'usd_value': 'sum',
        'currency': 'count'
    }).rename(columns={'currency': 'count'})
    summary['withdrawals_by_currency'] = withdrawals_by_currency.to_dict('index')

    # Date range
    all_dates = pd.concat([deposits_df['timestamp'], withdrawals_df['timestamp']])
    if len(all_dates) > 0:
        summary['first_transaction'] = all_dates.min()
        summary['last_transaction'] = all_dates.max()

    # Transactions with missing prices
    deposits_missing = len(deposits_df[deposits_df['usd_value'] == 0])
    withdrawals_missing = len(withdrawals_df[withdrawals_df['usd_value'] == 0])
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

    # -------------------------------------------------------------------------
    # 1. Deposits vs Withdrawals Over Time
    # -------------------------------------------------------------------------
    fig_timeline = go.Figure()

    # Group by month for cleaner visualization
    deposits_monthly = deposits_df.groupby(
        deposits_df['timestamp'].dt.to_period('M')
    )['usd_value'].sum()
    deposits_monthly.index = deposits_monthly.index.to_timestamp()

    withdrawals_monthly = withdrawals_df.groupby(
        withdrawals_df['timestamp'].dt.to_period('M')
    )['usd_value'].sum()
    withdrawals_monthly.index = withdrawals_monthly.index.to_timestamp()

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
    dep_by_curr = deposits_df.groupby('currency')['usd_value'].sum()
    fig_breakdown.add_trace(go.Pie(
        labels=dep_by_curr.index,
        values=dep_by_curr.values,
        hole=0.4,
        textinfo='label+percent',
        hovertemplate='%{label}: $%{value:,.2f}<extra></extra>',
    ), row=1, col=1)

    # Withdrawals pie
    wit_by_curr = withdrawals_df.groupby('currency')['usd_value'].sum()
    fig_breakdown.add_trace(go.Pie(
        labels=wit_by_curr.index,
        values=wit_by_curr.values,
        hole=0.4,
        textinfo='label+percent',
        hovertemplate='%{label}: $%{value:,.2f}<extra></extra>',
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
    all_txs = pd.concat([
        deposits_df[['timestamp', 'usd_value']].assign(type='deposit'),
        withdrawals_df[['timestamp', 'usd_value']].assign(type='withdrawal')
    ]).sort_values('timestamp')

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
