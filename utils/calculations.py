"""Statistics and metrics calculation utilities."""

import pandas as pd
import numpy as np
from typing import Dict, Any
import streamlit as st


@st.cache_data
def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics from betting data.

    Args:
        df: Preprocessed betting DataFrame

    Returns:
        Dictionary containing all calculated statistics
    """
    if len(df) == 0:
        return {}

    stats = {}

    # Overall statistics
    stats['total_bets'] = len(df)
    stats['total_wagered'] = df['bet_amount'].sum()
    stats['total_payouts'] = df['payout_amount'].sum()
    stats['net_profit'] = stats['total_payouts'] - stats['total_wagered']
    stats['profit_percentage'] = (
        (stats['net_profit'] / stats['total_wagered'] * 100)
        if stats['total_wagered'] > 0 else 0
    )
    stats['win_rate'] = df['is_win'].mean() * 100
    stats['total_wins'] = df['is_win'].sum()
    stats['total_losses'] = len(df) - stats['total_wins']

    # RTP (Return to Player)
    stats['rtp'] = (
        (stats['total_payouts'] / stats['total_wagered'] * 100)
        if stats['total_wagered'] > 0 else 0
    )

    # Bet statistics
    stats['avg_bet_size'] = df['bet_amount'].mean()
    stats['median_bet_size'] = df['bet_amount'].median()
    stats['max_bet_size'] = df['bet_amount'].max()
    stats['min_bet_size'] = df['bet_amount'].min()
    stats['std_bet_size'] = df['bet_amount'].std()

    # Win statistics
    wins = df[df['is_win']]
    if len(wins) > 0:
        stats['avg_multiplier'] = wins['multiplier'].mean()
        stats['max_multiplier'] = wins['multiplier'].max()
        stats['biggest_win'] = wins['payout_amount'].max()
        biggest_win_idx = wins['payout_amount'].idxmax()
        stats['biggest_win_multiplier'] = df.loc[biggest_win_idx, 'multiplier']
        stats['biggest_win_game'] = df.loc[biggest_win_idx, 'game']
    else:
        stats['avg_multiplier'] = 0
        stats['max_multiplier'] = 0
        stats['biggest_win'] = 0
        stats['biggest_win_multiplier'] = 0
        stats['biggest_win_game'] = 'N/A'

    # Biggest loss
    losses = df[~df['is_win']]
    stats['biggest_loss'] = losses['bet_amount'].max() if len(losses) > 0 else 0

    # Time range
    stats['first_bet'] = df['timestamp'].min()
    stats['last_bet'] = df['timestamp'].max()
    stats['days_active'] = (stats['last_bet'] - stats['first_bet']).days + 1
    stats['bets_per_day'] = stats['total_bets'] / max(stats['days_active'], 1)

    # Game breakdown
    game_stats = df.groupby('game').agg({
        'bet_amount': ['sum', 'mean', 'count'],
        'payout_amount': 'sum',
        'profit': 'sum',
        'is_win': 'mean',
        'multiplier': 'mean',
    }).round(4)
    game_stats.columns = ['total_wagered', 'avg_bet', 'num_bets',
                          'total_payouts', 'net_profit', 'win_rate', 'avg_mult']
    game_stats['rtp'] = (game_stats['total_payouts'] / game_stats['total_wagered'] * 100).round(2)
    stats['game_stats'] = game_stats.to_dict('index')

    # Best and worst game
    if len(game_stats) > 0:
        stats['best_game'] = game_stats['net_profit'].idxmax()
        stats['best_game_profit'] = game_stats['net_profit'].max()
        stats['worst_game'] = game_stats['net_profit'].idxmin()
        stats['worst_game_profit'] = game_stats['net_profit'].min()
    else:
        stats['best_game'] = 'N/A'
        stats['best_game_profit'] = 0
        stats['worst_game'] = 'N/A'
        stats['worst_game_profit'] = 0

    # Session statistics
    session_stats = df.groupby('session_id').agg({
        'bet_amount': 'sum',
        'profit': 'sum',
        'timestamp': ['min', 'max', 'count'],
    })
    session_stats.columns = ['wagered', 'profit', 'start', 'end', 'num_bets']
    session_stats['duration_mins'] = (
        (session_stats['end'] - session_stats['start']).dt.total_seconds() / 60
    )
    stats['total_sessions'] = len(session_stats)
    stats['avg_session_duration'] = session_stats['duration_mins'].mean()
    stats['avg_session_bets'] = session_stats['num_bets'].mean()
    stats['avg_session_profit'] = session_stats['profit'].mean()

    # Drawdown analysis
    cumulative = df['cumulative_profit']
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    stats['max_drawdown'] = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    if running_max[max_dd_idx] > 0:
        stats['max_drawdown_pct'] = (stats['max_drawdown'] / running_max[max_dd_idx] * 100)
    else:
        stats['max_drawdown_pct'] = 0

    # Streak analysis
    stats['streaks'] = calculate_streaks(df)

    # Behavioral analysis
    stats['behavior'] = analyze_betting_behavior(df)

    # Time patterns
    stats['hourly_stats'] = df.groupby('hour')['profit'].agg(['sum', 'mean', 'count']).to_dict()
    stats['daily_stats'] = df.groupby('day_of_week')['profit'].agg(['sum', 'mean', 'count']).to_dict()

    return stats


def calculate_streaks(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate winning and losing streak statistics."""
    if len(df) == 0:
        return {
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'avg_win_streak': 0,
            'avg_loss_streak': 0,
            'win_streaks': [],
            'loss_streaks': [],
        }

    streaks = {'win': [], 'loss': []}
    current_streak = 0
    current_type = None

    for is_win in df['is_win']:
        if is_win:
            if current_type == 'win':
                current_streak += 1
            else:
                if current_type == 'loss' and current_streak > 0:
                    streaks['loss'].append(current_streak)
                current_streak = 1
                current_type = 'win'
        else:
            if current_type == 'loss':
                current_streak += 1
            else:
                if current_type == 'win' and current_streak > 0:
                    streaks['win'].append(current_streak)
                current_streak = 1
                current_type = 'loss'

    # Don't forget the last streak
    if current_type and current_streak > 0:
        streaks[current_type].append(current_streak)

    return {
        'max_win_streak': max(streaks['win']) if streaks['win'] else 0,
        'max_loss_streak': max(streaks['loss']) if streaks['loss'] else 0,
        'avg_win_streak': np.mean(streaks['win']) if streaks['win'] else 0,
        'avg_loss_streak': np.mean(streaks['loss']) if streaks['loss'] else 0,
        'win_streaks': streaks['win'],
        'loss_streaks': streaks['loss'],
    }


def analyze_betting_behavior(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze betting behavior patterns."""
    behavior = {}

    if len(df) < 2:
        return behavior

    # Bet size after win vs loss
    df_shifted = df.copy()
    df_shifted['prev_win'] = df_shifted['is_win'].shift(1)
    df_shifted = df_shifted.dropna(subset=['prev_win'])

    bet_after_win = df_shifted[df_shifted['prev_win'] == True]['bet_amount']
    bet_after_loss = df_shifted[df_shifted['prev_win'] == False]['bet_amount']

    behavior['avg_bet_after_win'] = bet_after_win.mean() if len(bet_after_win) > 0 else 0
    behavior['avg_bet_after_loss'] = bet_after_loss.mean() if len(bet_after_loss) > 0 else 0

    if behavior['avg_bet_after_win'] > 0:
        behavior['loss_chase_ratio'] = (
            behavior['avg_bet_after_loss'] / behavior['avg_bet_after_win']
        )
    else:
        behavior['loss_chase_ratio'] = 1.0

    # Peak betting hours
    hour_counts = df.groupby('hour').size()
    behavior['peak_hour'] = hour_counts.idxmax() if len(hour_counts) > 0 else 0
    behavior['peak_hour_pct'] = (hour_counts.max() / len(df) * 100) if len(df) > 0 else 0

    # Night owl detection (10 PM - 4 AM)
    night_bets = df[df['hour'].isin([22, 23, 0, 1, 2, 3, 4])]
    behavior['night_betting_pct'] = len(night_bets) / len(df) * 100 if len(df) > 0 else 0

    return behavior


def get_session_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get aggregated session data."""
    if len(df) == 0:
        return pd.DataFrame()

    session_data = df.groupby('session_id').agg({
        'bet_amount': 'sum',
        'profit': 'sum',
        'game': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'mixed',
        'timestamp': ['min', 'max', 'count'],
    })
    session_data.columns = ['wagered', 'profit', 'main_game', 'start', 'end', 'num_bets']
    session_data['duration_mins'] = (
        (session_data['end'] - session_data['start']).dt.total_seconds() / 60
    )
    session_data = session_data.reset_index()

    return session_data
