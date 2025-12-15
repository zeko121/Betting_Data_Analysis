"""Data loading and preprocessing utilities."""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
import streamlit as st


@st.cache_data
def load_json_data(file_contents: Union[bytes, str]) -> pd.DataFrame:
    """
    Load betting data from JSON content.

    Args:
        file_contents: Raw JSON content from uploaded file

    Returns:
        DataFrame with raw betting data
    """
    if isinstance(file_contents, bytes):
        file_contents = file_contents.decode('utf-8')

    data = json.loads(file_contents)

    # Handle both array format and object with 'bets' key
    if isinstance(data, list):
        bets = data
    elif isinstance(data, dict):
        bets = data.get('bets', data.get('data', [data]))
    else:
        raise ValueError("Unexpected data format")

    if not bets:
        raise ValueError("No bets found in file")

    return pd.DataFrame(bets)


@st.cache_data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw betting data into analysis-ready format.

    Args:
        df: Raw DataFrame from load_json_data

    Returns:
        Processed DataFrame with extracted and calculated fields
    """
    processed = df.copy()

    # Extract nested 'data' fields if present
    if 'data' in processed.columns and processed['data'].dtype == 'object':
        data_df = pd.json_normalize(processed['data'])
        for col in data_df.columns:
            if col not in processed.columns:
                processed[col] = data_df[col].values

    # Ensure required columns exist with defaults
    required_cols = {
        'amount': 0.0,
        'payout': 0.0,
        'payoutMultiplier': 0.0,
        'currency': 'usdt',
        'gameName': 'unknown',
        'createdAt': None,
    }

    for col, default in required_cols.items():
        if col not in processed.columns:
            processed[col] = default

    # Convert timestamps
    if 'createdAt' in processed.columns:
        processed['timestamp'] = pd.to_datetime(
            processed['createdAt'],
            unit='ms',
            errors='coerce'
        )
    elif 'created_at' in processed.columns:
        processed['timestamp'] = pd.to_datetime(
            processed['created_at'],
            errors='coerce'
        )
    else:
        processed['timestamp'] = pd.NaT

    # Fill missing timestamps
    processed['timestamp'] = processed['timestamp'].fillna(pd.Timestamp.now())

    # Calculate derived fields
    processed['profit'] = processed['payout'] - processed['amount']
    processed['is_win'] = processed['payout'] > 0
    processed['bet_amount'] = processed['amount'].astype(float)
    processed['payout_amount'] = processed['payout'].astype(float)
    processed['multiplier'] = processed['payoutMultiplier'].astype(float)

    # Normalize game names
    processed['game'] = processed['gameName'].str.lower().str.strip()
    processed['game'] = processed['game'].replace({
        'coin flip': 'flip',
        'coinflip': 'flip',
        'coin-flip': 'flip',
    })

    # Extract time components
    processed['date'] = processed['timestamp'].dt.date
    processed['datetime'] = processed['timestamp']
    processed['hour'] = processed['timestamp'].dt.hour
    processed['day_of_week'] = processed['timestamp'].dt.day_name()
    processed['week'] = processed['timestamp'].dt.isocalendar().week
    processed['month'] = processed['timestamp'].dt.month
    processed['year'] = processed['timestamp'].dt.year

    # Sort by timestamp
    processed = processed.sort_values('timestamp').reset_index(drop=True)

    # Calculate cumulative profit
    processed['cumulative_profit'] = processed['profit'].cumsum()

    # Calculate rolling statistics
    processed['rolling_win_rate'] = (
        processed['is_win']
        .rolling(window=100, min_periods=1)
        .mean()
    )

    # Identify sessions (gaps > 30 minutes)
    time_diff = processed['timestamp'].diff()
    session_break = time_diff > pd.Timedelta(minutes=30)
    processed['session_id'] = session_break.cumsum()

    # Extract game-specific state data
    processed = extract_game_states(processed)

    return processed


def extract_game_states(df: pd.DataFrame) -> pd.DataFrame:
    """Extract game-specific state information (Plinko, Keno, etc.)"""

    # Plinko state
    if 'statePlinko' in df.columns:
        plinko_data = df['statePlinko'].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
        df['plinko_risk'] = plinko_data.apply(lambda x: x.get('risk', None))
        df['plinko_rows'] = plinko_data.apply(lambda x: x.get('rows', None))
        # Drop the original dict column to avoid hashing issues
        df = df.drop(columns=['statePlinko'], errors='ignore')

    # Keno state
    if 'stateKeno' in df.columns:
        keno_data = df['stateKeno'].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
        df['keno_risk'] = keno_data.apply(lambda x: x.get('risk', None))
        # Convert lists to strings for hashability
        df['keno_selected'] = keno_data.apply(
            lambda x: ','.join(map(str, x.get('selectedNumbers', [])))
        )
        df['keno_drawn'] = keno_data.apply(
            lambda x: ','.join(map(str, x.get('drawnNumbers', [])))
        )
        df['keno_num_selected'] = keno_data.apply(
            lambda x: len(x.get('selectedNumbers', []))
        )
        # Drop the original dict column to avoid hashing issues
        df = df.drop(columns=['stateKeno'], errors='ignore')

    # Drop any remaining columns that might contain dicts or lists
    cols_to_drop = []
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains dicts or lists
            sample = df[col].dropna().head(1)
            if len(sample) > 0:
                val = sample.iloc[0]
                if isinstance(val, (dict, list)):
                    cols_to_drop.append(col)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors='ignore')

    return df


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple = None,
    selected_games: List[str] = None,
    bet_range: tuple = None,
    result_filter: str = "all"
) -> pd.DataFrame:
    """
    Apply filters to the betting data.

    Args:
        df: Preprocessed DataFrame
        date_range: Tuple of (start_date, end_date)
        selected_games: List of game names to include
        bet_range: Tuple of (min_bet, max_bet)
        result_filter: "all", "wins", or "losses"

    Returns:
        Filtered DataFrame
    """
    filtered = df.copy()

    # Date filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            filtered = filtered[
                (filtered['date'] >= start_date) &
                (filtered['date'] <= end_date)
            ]

    # Game filter
    if selected_games:
        filtered = filtered[filtered['game'].isin(selected_games)]

    # Bet size filter
    if bet_range and len(bet_range) == 2:
        min_bet, max_bet = bet_range
        filtered = filtered[
            (filtered['bet_amount'] >= min_bet) &
            (filtered['bet_amount'] <= max_bet)
        ]

    # Result filter
    if result_filter == "wins":
        filtered = filtered[filtered['is_win'] == True]
    elif result_filter == "losses":
        filtered = filtered[filtered['is_win'] == False]

    # Recalculate cumulative profit for filtered data
    if len(filtered) > 0:
        filtered = filtered.sort_values('timestamp').reset_index(drop=True)
        filtered['cumulative_profit'] = filtered['profit'].cumsum()

    return filtered
