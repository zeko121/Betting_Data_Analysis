#!/usr/bin/env python3
"""
Betting Data Analysis & Visualization System

A comprehensive dashboard for analyzing crypto casino betting history.
Generates interactive HTML visualizations and static PNG exports.

Usage:
    python betting_visualizer.py bet_archive.json
    python betting_visualizer.py file1.json file2.json -o dashboard.html --export-png
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PNG export
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# COLOR PALETTE & STYLING
# =============================================================================

COLORS = {
    'background': '#1a1a2e',
    'card_bg': '#16213e',
    'profit': '#00d4aa',
    'loss': '#ff4757',
    'neutral': '#5dade2',
    'secondary': '#a55eea',
    'text': '#ffffff',
    'muted': '#888888',
    'grid': '#2d2d44',
    'accent1': '#feca57',
    'accent2': '#ff9ff3',
    'accent3': '#54a0ff',
}

PLOTLY_LAYOUT = {
    'paper_bgcolor': COLORS['background'],
    'plot_bgcolor': COLORS['background'],
    'font': {'color': COLORS['text'], 'family': 'Arial, sans-serif'},
    'xaxis': {
        'gridcolor': COLORS['grid'],
        'gridwidth': 0.5,
        'griddash': 'dot',
        'zerolinecolor': COLORS['grid'],
    },
    'yaxis': {
        'gridcolor': COLORS['grid'],
        'gridwidth': 0.5,
        'griddash': 'dot',
        'zerolinecolor': COLORS['grid'],
    },
    'legend': {
        'bgcolor': 'rgba(0,0,0,0)',
        'bordercolor': 'rgba(0,0,0,0)',
    },
    'margin': {'l': 60, 'r': 40, 't': 60, 'b': 60},
}


def apply_dark_theme(fig: go.Figure, title: str = None, **kwargs) -> go.Figure:
    """Apply dark theme styling to a Plotly figure."""
    layout_update = {**PLOTLY_LAYOUT, **kwargs}
    if title:
        layout_update['title'] = {'text': title, 'font': {'size': 18, 'color': COLORS['text']}}
    fig.update_layout(**layout_update)
    return fig

GAME_COLORS = {
    'plinko': '#00d4aa',
    'keno': '#ff9ff3',
    'Flip': '#feca57',
    'dice': '#54a0ff',
    'limbo': '#ff4757',
    'mines': '#a55eea',
    'crash': '#ff6b6b',
    'blackjack': '#2ed573',
    'roulette': '#ffa502',
    'slots': '#3742fa',
    'sportsbook': '#70a1ff',
    'other': '#888888',
}


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def load_data(filepaths: List[str]) -> pd.DataFrame:
    """
    Load betting data from one or more JSON files.

    Args:
        filepaths: List of paths to JSON files containing bet data

    Returns:
        DataFrame with all bets combined
    """
    all_bets = []

    for filepath in filepaths:
        print(f"Loading {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle both array format and object with 'bets' key
            if isinstance(data, list):
                bets = data
            elif isinstance(data, dict):
                bets = data.get('bets', data.get('data', [data]))
            else:
                print(f"Warning: Unexpected data format in {filepath}")
                continue

            all_bets.extend(bets)
            print(f"  Loaded {len(bets)} bets")

        except json.JSONDecodeError as e:
            print(f"Error parsing {filepath}: {e}")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    if not all_bets:
        raise ValueError("No bets loaded from any file")

    print(f"Total bets loaded: {len(all_bets)}")
    return pd.DataFrame(all_bets)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw betting data into analysis-ready format.

    Args:
        df: Raw DataFrame from load_data

    Returns:
        Processed DataFrame with extracted and calculated fields
    """
    processed = df.copy()

    # Extract nested 'data' fields if present
    if 'data' in processed.columns and processed['data'].dtype == 'object':
        data_df = pd.json_normalize(processed['data'])
        # Prefix nested columns to avoid conflicts
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
        # Handle millisecond timestamps
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

    # Fill missing timestamps with current time (edge case)
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

    # Keno state
    if 'stateKeno' in df.columns:
        keno_data = df['stateKeno'].apply(
            lambda x: x if isinstance(x, dict) else {}
        )
        df['keno_risk'] = keno_data.apply(lambda x: x.get('risk', None))
        df['keno_selected'] = keno_data.apply(
            lambda x: x.get('selectedNumbers', [])
        )
        df['keno_drawn'] = keno_data.apply(
            lambda x: x.get('drawnNumbers', [])
        )
        df['keno_num_selected'] = df['keno_selected'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

    return df


# =============================================================================
# STATISTICS CALCULATION
# =============================================================================

def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics from betting data.

    Args:
        df: Preprocessed betting DataFrame

    Returns:
        Dictionary containing all calculated statistics
    """
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
    stats['biggest_loss'] = df[~df['is_win']]['bet_amount'].max() if len(df[~df['is_win']]) > 0 else 0

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
    stats['max_drawdown_pct'] = (
        (stats['max_drawdown'] / running_max[drawdown.idxmin()] * 100)
        if running_max[drawdown.idxmin()] > 0 else 0
    )

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
    behavior['peak_hour'] = hour_counts.idxmax()
    behavior['peak_hour_pct'] = hour_counts.max() / len(df) * 100

    # Night owl detection (10 PM - 4 AM)
    night_bets = df[df['hour'].isin([22, 23, 0, 1, 2, 3, 4])]
    behavior['night_betting_pct'] = len(night_bets) / len(df) * 100

    return behavior


# =============================================================================
# VISUALIZATION FUNCTIONS - FINANCIAL OVERVIEW
# =============================================================================

def create_cumulative_pnl_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create cumulative profit/loss over time chart.
    The most important chart - shows the overall journey.
    """
    fig = go.Figure()

    # Create color array based on profit value
    colors = [COLORS['profit'] if p >= 0 else COLORS['loss']
              for p in df['cumulative_profit']]

    # Main line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cumulative_profit'],
        mode='lines',
        name='Cumulative P&L',
        line=dict(color=COLORS['profit'], width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.1)',
        hovertemplate='<b>%{x}</b><br>P&L: $%{y:.2f}<extra></extra>',
    ))

    # Zero line
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color=COLORS['muted'],
        annotation_text='Break Even',
        annotation_position='right',
    )

    # Add markers for significant events
    max_profit_idx = df['cumulative_profit'].idxmax()
    min_profit_idx = df['cumulative_profit'].idxmin()

    fig.add_trace(go.Scatter(
        x=[df.loc[max_profit_idx, 'timestamp']],
        y=[df.loc[max_profit_idx, 'cumulative_profit']],
        mode='markers',
        name='Peak',
        marker=dict(color=COLORS['profit'], size=12, symbol='star'),
        hovertemplate='Peak: $%{y:.2f}<extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=[df.loc[min_profit_idx, 'timestamp']],
        y=[df.loc[min_profit_idx, 'cumulative_profit']],
        mode='markers',
        name='Trough',
        marker=dict(color=COLORS['loss'], size=12, symbol='star'),
        hovertemplate='Trough: $%{y:.2f}<extra></extra>',
    ))

    apply_dark_theme(
        fig,
        title='Cumulative Profit/Loss Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative P&L (USDT)',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    return fig


def create_pnl_by_game_chart(df: pd.DataFrame, stats: Dict) -> go.Figure:
    """Create horizontal bar chart showing P&L by game."""
    game_profits = df.groupby('game')['profit'].sum().sort_values()

    colors = [COLORS['profit'] if p >= 0 else COLORS['loss'] for p in game_profits.values]

    fig = go.Figure(go.Bar(
        x=game_profits.values,
        y=game_profits.index,
        orientation='h',
        marker_color=colors,
        text=[f'${p:+.2f}' for p in game_profits.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>P&L: $%{x:.2f}<extra></extra>',
    ))

    fig.add_vline(x=0, line_dash='dash', line_color=COLORS['muted'])

    apply_dark_theme(
        fig,
        title='Profit/Loss by Game',
        xaxis_title='Net Profit/Loss (USDT)',
        yaxis_title='Game',
    )

    return fig


def create_wagered_vs_payout_chart(df: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart comparing wagered vs payouts by game."""
    game_data = df.groupby('game').agg({
        'bet_amount': 'sum',
        'payout_amount': 'sum',
    }).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Wagered',
        x=game_data['game'],
        y=game_data['bet_amount'],
        marker_color=COLORS['neutral'],
        hovertemplate='<b>%{x}</b><br>Wagered: $%{y:.2f}<extra></extra>',
    ))

    fig.add_trace(go.Bar(
        name='Payouts',
        x=game_data['game'],
        y=game_data['payout_amount'],
        marker_color=COLORS['secondary'],
        hovertemplate='<b>%{x}</b><br>Payouts: $%{y:.2f}<extra></extra>',
    ))

    apply_dark_theme(
        fig,
        title='Total Wagered vs Total Payouts by Game',
        xaxis_title='Game',
        yaxis_title='Amount (USDT)',
        barmode='group',
    )

    return fig


def create_daily_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create calendar heatmap showing daily performance."""
    daily_profit = df.groupby('date')['profit'].sum().reset_index()
    daily_profit['date'] = pd.to_datetime(daily_profit['date'])
    daily_profit['week'] = daily_profit['date'].dt.isocalendar().week
    daily_profit['day_of_week'] = daily_profit['date'].dt.dayofweek
    daily_profit['month_year'] = daily_profit['date'].dt.strftime('%Y-%m')

    # Create pivot table for heatmap
    pivot = daily_profit.pivot_table(
        index='day_of_week',
        columns='week',
        values='profit',
        aggfunc='sum'
    )

    # Custom colorscale: red for loss, green for profit
    max_val = max(abs(pivot.min().min()), abs(pivot.max().max()))

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        colorscale=[
            [0, COLORS['loss']],
            [0.5, COLORS['background']],
            [1, COLORS['profit']]
        ],
        zmid=0,
        hovertemplate='Week %{x}<br>%{y}<br>P&L: $%{z:.2f}<extra></extra>',
        colorbar=dict(
            title='P&L',
            tickformat='$.2f',
        ),
    ))

    apply_dark_theme(
        fig,
        title='Daily Profit/Loss Heatmap',
        xaxis_title='Week Number',
        yaxis_title='Day of Week',
        )

    return fig


# =============================================================================
# VISUALIZATION FUNCTIONS - BETTING BEHAVIOR
# =============================================================================

def create_bet_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create histogram showing bet size distribution."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Bet Size Distribution', 'Bet Size by Game')
    )

    # Overall distribution
    fig.add_trace(go.Histogram(
        x=df['bet_amount'],
        nbinsx=50,
        marker_color=COLORS['neutral'],
        opacity=0.7,
        name='All Bets',
        hovertemplate='Bet Size: $%{x:.2f}<br>Count: %{y}<extra></extra>',
    ), row=1, col=1)

    # Add mean and median lines
    mean_bet = df['bet_amount'].mean()
    median_bet = df['bet_amount'].median()

    fig.add_vline(
        x=mean_bet,
        line_dash='dash',
        line_color=COLORS['profit'],
        annotation_text=f'Mean: ${mean_bet:.2f}',
        row=1, col=1,
    )
    fig.add_vline(
        x=median_bet,
        line_dash='dot',
        line_color=COLORS['secondary'],
        annotation_text=f'Median: ${median_bet:.2f}',
        row=1, col=1,
    )

    # Distribution by game (box plot)
    games = df['game'].unique()
    for game in games:
        game_data = df[df['game'] == game]['bet_amount']
        color = GAME_COLORS.get(game, COLORS['muted'])
        fig.add_trace(go.Box(
            y=game_data,
            name=game.title(),
            marker_color=color,
            boxpoints='outliers',
        ), row=1, col=2)

    apply_dark_theme(
        fig,
        title='Bet Size Distribution Analysis',
        showlegend=False,
        )

    fig.update_xaxes(title_text='Bet Amount (USDT)', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_xaxes(title_text='Game', row=1, col=2)
    fig.update_yaxes(title_text='Bet Amount (USDT)', row=1, col=2)

    return fig


def create_betting_frequency_chart(df: pd.DataFrame) -> go.Figure:
    """Create stacked area chart showing betting frequency over time."""
    # Group by date and game
    daily_counts = df.groupby(['date', 'game']).size().unstack(fill_value=0)

    fig = go.Figure()

    games = daily_counts.columns.tolist()
    for game in games:
        color = GAME_COLORS.get(game, COLORS['muted'])
        fig.add_trace(go.Scatter(
            x=daily_counts.index,
            y=daily_counts[game],
            mode='lines',
            name=game.title(),
            stackgroup='one',
            fillcolor=color,
            line=dict(color=color, width=0.5),
            hovertemplate=f'<b>{game.title()}</b><br>Date: %{{x}}<br>Bets: %{{y}}<extra></extra>',
        ))

    apply_dark_theme(
        fig,
        title='Betting Frequency Over Time (by Game)',
        xaxis_title='Date',
        yaxis_title='Number of Bets',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    return fig


def create_session_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot analyzing betting sessions."""
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
    session_data = session_data[session_data['duration_mins'] > 0]  # Filter zero-duration

    # Color by main game
    colors = [GAME_COLORS.get(g, COLORS['muted']) for g in session_data['main_game']]

    fig = go.Figure(go.Scatter(
        x=session_data['duration_mins'],
        y=session_data['profit'],
        mode='markers',
        marker=dict(
            size=np.clip(session_data['num_bets'] / 2, 5, 50),
            color=colors,
            opacity=0.7,
            line=dict(width=1, color=COLORS['text']),
        ),
        text=session_data['main_game'],
        hovertemplate=(
            '<b>Session</b><br>'
            'Duration: %{x:.1f} mins<br>'
            'P&L: $%{y:.2f}<br>'
            'Game: %{text}<br>'
            '<extra></extra>'
        ),
    ))

    fig.add_hline(y=0, line_dash='dash', line_color=COLORS['muted'])

    # Add trendline
    if len(session_data) > 2:
        z = np.polyfit(session_data['duration_mins'], session_data['profit'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(session_data['duration_mins'].min(),
                             session_data['duration_mins'].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            name='Trend',
            line=dict(color=COLORS['secondary'], dash='dash'),
        ))

    apply_dark_theme(
        fig,
        title='Session Analysis: Duration vs Profit',
        xaxis_title='Session Duration (minutes)',
        yaxis_title='Session Profit/Loss (USDT)',
    )

    return fig


def create_hourly_pattern_chart(df: pd.DataFrame) -> go.Figure:
    """Create polar/radar chart showing betting patterns by hour."""
    hourly = df.groupby('hour').agg({
        'profit': ['sum', 'mean'],
        'bet_amount': 'count',
    })
    hourly.columns = ['total_profit', 'avg_profit', 'num_bets']
    hourly = hourly.reindex(range(24), fill_value=0)

    # Create polar chart
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'polar'}, {'type': 'polar'}]],
        subplot_titles=('Bets by Hour', 'Avg Profit by Hour')
    )

    # Bets by hour
    fig.add_trace(go.Barpolar(
        r=hourly['num_bets'],
        theta=[f'{h}:00' for h in range(24)],
        marker_color=COLORS['neutral'],
        marker_line_color=COLORS['text'],
        marker_line_width=1,
        opacity=0.8,
        name='Bet Count',
    ), row=1, col=1)

    # Profit by hour
    colors = [COLORS['profit'] if p >= 0 else COLORS['loss'] for p in hourly['avg_profit']]
    fig.add_trace(go.Barpolar(
        r=abs(hourly['avg_profit']),
        theta=[f'{h}:00' for h in range(24)],
        marker_color=colors,
        marker_line_color=COLORS['text'],
        marker_line_width=1,
        opacity=0.8,
        name='Avg Profit',
    ), row=1, col=2)

    apply_dark_theme(
        fig,
        title='Betting Patterns by Hour of Day',
        polar=dict(bgcolor=COLORS['background']),
        polar2=dict(bgcolor=COLORS['background']),
    )

    return fig


def create_day_of_week_chart(df: pd.DataFrame) -> go.Figure:
    """Create box plot showing P&L distribution by day of week."""
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Calculate daily totals
    daily_totals = df.groupby(['date', 'day_of_week'])['profit'].sum().reset_index()

    fig = go.Figure()

    for day in day_order:
        day_data = daily_totals[daily_totals['day_of_week'] == day]['profit']
        if len(day_data) > 0:
            color = COLORS['profit'] if day_data.mean() >= 0 else COLORS['loss']
            fig.add_trace(go.Box(
                y=day_data,
                name=day[:3],
                marker_color=color,
                boxpoints='outliers',
            ))

    fig.add_hline(y=0, line_dash='dash', line_color=COLORS['muted'])

    apply_dark_theme(
        fig,
        title='Daily Profit/Loss Distribution by Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Daily P&L (USDT)',
    )

    return fig


# =============================================================================
# VISUALIZATION FUNCTIONS - MULTIPLIER & WINS ANALYSIS
# =============================================================================

def create_multiplier_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create histogram showing distribution of winning multipliers."""
    wins = df[df['is_win'] & (df['multiplier'] > 0)]

    if len(wins) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No winning bets to display",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Multiplier Distribution (Log Scale)', 'Top Multipliers by Game')
    )

    # Log-scale histogram
    fig.add_trace(go.Histogram(
        x=np.log10(wins['multiplier'] + 0.1),
        nbinsx=50,
        marker_color=COLORS['profit'],
        opacity=0.7,
        name='Multipliers',
        hovertemplate='Log10(Mult): %{x:.2f}<br>Count: %{y}<extra></extra>',
    ), row=1, col=1)

    # Add annotations for biggest multipliers
    top_5 = wins.nlargest(5, 'multiplier')
    for _, row in top_5.iterrows():
        fig.add_vline(
            x=np.log10(row['multiplier']),
            line_dash='dot',
            line_color=COLORS['accent1'],
            annotation_text=f"{row['multiplier']:.1f}x",
            row=1, col=1,
        )

    # Multiplier distribution by game
    for game in wins['game'].unique():
        game_data = wins[wins['game'] == game]['multiplier']
        color = GAME_COLORS.get(game, COLORS['muted'])
        fig.add_trace(go.Box(
            y=game_data,
            name=game.title(),
            marker_color=color,
            boxpoints='outliers',
        ), row=1, col=2)

    apply_dark_theme(
        fig,
        title='Winning Multiplier Distribution',
        )

    fig.update_xaxes(title_text='Log10(Multiplier)', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_yaxes(title_text='Multiplier', row=1, col=2)

    return fig


def create_big_wins_timeline_chart(df: pd.DataFrame, threshold_mult: float = 5.0) -> go.Figure:
    """Create scatter plot showing big wins over time."""
    big_wins = df[(df['is_win']) & (df['multiplier'] >= threshold_mult)]

    if len(big_wins) == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No wins above {threshold_mult}x multiplier",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    colors = [GAME_COLORS.get(g, COLORS['muted']) for g in big_wins['game']]

    fig = go.Figure(go.Scatter(
        x=big_wins['timestamp'],
        y=big_wins['payout_amount'],
        mode='markers',
        marker=dict(
            size=np.clip(big_wins['multiplier'] / 2, 8, 40),
            color=colors,
            opacity=0.8,
            line=dict(width=1, color=COLORS['text']),
        ),
        text=[f"{g.title()} - {m:.1f}x" for g, m in zip(big_wins['game'], big_wins['multiplier'])],
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Date: %{x}<br>'
            'Payout: $%{y:.2f}<br>'
            '<extra></extra>'
        ),
    ))

    # Add annotations for top 3 wins
    top_3 = big_wins.nlargest(3, 'payout_amount')
    for _, row in top_3.iterrows():
        fig.add_annotation(
            x=row['timestamp'],
            y=row['payout_amount'],
            text=f"${row['payout_amount']:.2f} ({row['multiplier']:.1f}x)",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLORS['text'],
            font=dict(color=COLORS['text'], size=10),
        )

    apply_dark_theme(
        fig,
        title=f'Big Wins Timeline (≥{threshold_mult}x Multiplier)',
        xaxis_title='Date',
        yaxis_title='Payout Amount (USDT)',
    )

    return fig


def create_win_rate_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Create rolling win rate chart over time."""
    fig = go.Figure()

    # Overall rolling win rate
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['rolling_win_rate'] * 100,
        mode='lines',
        name='Rolling Win Rate (100 bets)',
        line=dict(color=COLORS['neutral'], width=2),
        hovertemplate='%{x}<br>Win Rate: %{y:.1f}%<extra></extra>',
    ))

    # Overall average line
    avg_win_rate = df['is_win'].mean() * 100
    fig.add_hline(
        y=avg_win_rate,
        line_dash='dash',
        line_color=COLORS['profit'],
        annotation_text=f'Average: {avg_win_rate:.1f}%',
    )

    # Add 50% reference line
    fig.add_hline(y=50, line_dash='dot', line_color=COLORS['muted'])

    apply_dark_theme(
        fig,
        title='Rolling Win Rate Over Time',
        xaxis_title='Date',
        yaxis_title='Win Rate (%)',
        yaxis=dict(range=[0, 100]),
    )

    return fig


def create_risk_reward_scatter(df: pd.DataFrame) -> go.Figure:
    """Create scatter plot of risk vs reward by session."""
    session_data = df.groupby('session_id').agg({
        'bet_amount': 'sum',
        'profit': 'sum',
        'game': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'mixed',
    }).reset_index()
    session_data.columns = ['session_id', 'wagered', 'profit', 'main_game']

    colors = [GAME_COLORS.get(g, COLORS['muted']) for g in session_data['main_game']]

    fig = go.Figure(go.Scatter(
        x=session_data['wagered'],
        y=session_data['profit'],
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            opacity=0.7,
            line=dict(width=1, color=COLORS['text']),
        ),
        text=session_data['main_game'],
        hovertemplate=(
            '<b>Session</b><br>'
            'Wagered: $%{x:.2f}<br>'
            'Profit: $%{y:.2f}<br>'
            'Game: %{text}<br>'
            '<extra></extra>'
        ),
    ))

    # Add zero profit line
    fig.add_hline(y=0, line_dash='dash', line_color=COLORS['muted'])

    # Add trendline
    if len(session_data) > 2:
        z = np.polyfit(session_data['wagered'], session_data['profit'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, session_data['wagered'].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            name='Trend',
            line=dict(color=COLORS['secondary'], dash='dash'),
        ))

    apply_dark_theme(
        fig,
        title='Risk vs Reward by Session',
        xaxis_title='Total Wagered (USDT)',
        yaxis_title='Session Profit/Loss (USDT)',
    )

    return fig


def create_multiplier_bet_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create 2D histogram showing multiplier vs bet size."""
    wins = df[df['is_win'] & (df['multiplier'] > 0)]

    if len(wins) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No winning bets to display",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = go.Figure(go.Histogram2d(
        x=wins['bet_amount'],
        y=wins['multiplier'],
        colorscale=[
            [0, COLORS['background']],
            [0.5, COLORS['neutral']],
            [1, COLORS['profit']]
        ],
        nbinsx=30,
        nbinsy=30,
        colorbar=dict(title='Frequency'),
        hovertemplate='Bet: $%{x:.2f}<br>Mult: %{y:.1f}x<br>Count: %{z}<extra></extra>',
    ))

    apply_dark_theme(
        fig,
        title='Bet Size vs Multiplier Heatmap',
        xaxis_title='Bet Amount (USDT)',
        yaxis_title='Multiplier',
    )

    return fig


# =============================================================================
# VISUALIZATION FUNCTIONS - GAME-SPECIFIC DEEP DIVES
# =============================================================================

def create_plinko_analysis(df: pd.DataFrame) -> go.Figure:
    """Create Plinko-specific analysis charts."""
    plinko = df[df['game'] == 'plinko']

    if len(plinko) == 0 or 'plinko_risk' not in plinko.columns:
        fig = go.Figure()
        fig.add_annotation(text="No Plinko data available",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Risk Level Distribution',
            'Rows Preference',
            'P&L by Risk Level',
            'Multiplier by Risk Level'
        ),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'violin'}]]
    )

    # Risk level pie chart
    risk_counts = plinko['plinko_risk'].value_counts()
    fig.add_trace(go.Pie(
        values=risk_counts.values,
        labels=risk_counts.index,
        marker_colors=[COLORS['profit'], COLORS['accent1'], COLORS['loss']],
        hole=0.4,
    ), row=1, col=1)

    # Rows bar chart
    if 'plinko_rows' in plinko.columns:
        rows_counts = plinko['plinko_rows'].value_counts().sort_index()
        fig.add_trace(go.Bar(
            x=rows_counts.index.astype(str),
            y=rows_counts.values,
            marker_color=COLORS['neutral'],
        ), row=1, col=2)

    # P&L by risk level
    risk_profit = plinko.groupby('plinko_risk')['profit'].sum()
    colors = [COLORS['profit'] if p >= 0 else COLORS['loss'] for p in risk_profit.values]
    fig.add_trace(go.Bar(
        x=risk_profit.index,
        y=risk_profit.values,
        marker_color=colors,
        text=[f'${p:+.2f}' for p in risk_profit.values],
        textposition='outside',
    ), row=2, col=1)

    # Multiplier distribution by risk (violin plot)
    wins = plinko[plinko['is_win']]
    for risk in wins['plinko_risk'].unique():
        risk_data = wins[wins['plinko_risk'] == risk]['multiplier']
        if len(risk_data) > 0:
            fig.add_trace(go.Violin(
                y=risk_data,
                name=str(risk),
                box_visible=True,
                meanline_visible=True,
            ), row=2, col=2)

    apply_dark_theme(
        fig,
        title='Plinko Deep Dive Analysis',
        showlegend=False,
        )

    return fig


def create_keno_analysis(df: pd.DataFrame) -> go.Figure:
    """Create Keno-specific analysis charts."""
    keno = df[df['game'] == 'keno']

    if len(keno) == 0 or 'keno_selected' not in keno.columns:
        fig = go.Figure()
        fig.add_annotation(text="No Keno data available",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Numbers Selected Frequency',
            'Numbers Hit Frequency',
            'Hit Rate by Numbers Selected',
            'P&L by Numbers Selected'
        )
    )

    # Count number frequencies
    selected_counts = {}
    hit_counts = {}

    for _, row in keno.iterrows():
        selected = row.get('keno_selected', [])
        drawn = row.get('keno_drawn', [])

        if isinstance(selected, list):
            for num in selected:
                selected_counts[num] = selected_counts.get(num, 0) + 1

        if isinstance(drawn, list) and isinstance(selected, list):
            for num in drawn:
                if num in selected:
                    hit_counts[num] = hit_counts.get(num, 0) + 1

    # Selected numbers frequency
    if selected_counts:
        nums = sorted(selected_counts.keys())
        counts = [selected_counts[n] for n in nums]
        fig.add_trace(go.Bar(
            x=[str(n) for n in nums],
            y=counts,
            marker_color=COLORS['neutral'],
        ), row=1, col=1)

    # Hit numbers frequency
    if hit_counts:
        nums = sorted(hit_counts.keys())
        counts = [hit_counts[n] for n in nums]
        fig.add_trace(go.Bar(
            x=[str(n) for n in nums],
            y=counts,
            marker_color=COLORS['profit'],
        ), row=1, col=2)

    # Performance by number of selections
    if 'keno_num_selected' in keno.columns:
        by_selected = keno.groupby('keno_num_selected').agg({
            'is_win': 'mean',
            'profit': 'sum',
        }).reset_index()

        fig.add_trace(go.Scatter(
            x=by_selected['keno_num_selected'],
            y=by_selected['is_win'] * 100,
            mode='lines+markers',
            marker_color=COLORS['secondary'],
        ), row=2, col=1)

        colors = [COLORS['profit'] if p >= 0 else COLORS['loss'] for p in by_selected['profit']]
        fig.add_trace(go.Bar(
            x=by_selected['keno_num_selected'],
            y=by_selected['profit'],
            marker_color=colors,
        ), row=2, col=2)

    apply_dark_theme(
        fig,
        title='Keno Deep Dive Analysis',
        showlegend=False,
        )

    return fig


def create_game_comparison_radar(df: pd.DataFrame, stats: Dict) -> go.Figure:
    """Create radar chart comparing all games across multiple metrics."""
    game_stats = stats.get('game_stats', {})

    if not game_stats:
        fig = go.Figure()
        fig.add_annotation(text="No game statistics available",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Prepare data for radar chart
    categories = ['Win Rate', 'Avg Multiplier', 'RTP', 'Bet Frequency', 'Avg Bet']

    fig = go.Figure()

    for game, gs in game_stats.items():
        # Normalize values to 0-100 scale for radar
        values = [
            gs.get('win_rate', 0) * 100,  # Already 0-1
            min(gs.get('avg_mult', 0) * 10, 100),  # Scale multiplier
            min(gs.get('rtp', 0), 150),  # RTP can exceed 100
            min(gs.get('num_bets', 0) / max(df['game'].value_counts().max(), 1) * 100, 100),
            min(gs.get('avg_bet', 0) / max(df['bet_amount'].max(), 1) * 100, 100),
        ]
        values.append(values[0])  # Close the polygon

        color = GAME_COLORS.get(game, COLORS['muted'])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
            line=dict(color=color, width=2),
            name=game.title(),
        ))

    apply_dark_theme(
        fig,
        title='Game Comparison Radar',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=COLORS['grid'],
            ),
            angularaxis=dict(gridcolor=COLORS['grid']),
            bgcolor=COLORS['background'],
        ),
    )

    return fig


# =============================================================================
# VISUALIZATION FUNCTIONS - RISK & BANKROLL ANALYSIS
# =============================================================================

def create_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """Create maximum drawdown analysis chart."""
    cumulative = df['cumulative_profit']
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cumulative P&L with Peak', 'Drawdown from Peak'),
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
    )

    # Cumulative profit with running max
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=cumulative,
        mode='lines',
        name='Cumulative P&L',
        line=dict(color=COLORS['profit'], width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=running_max,
        mode='lines',
        name='Running Peak',
        line=dict(color=COLORS['accent1'], width=1, dash='dash'),
    ), row=1, col=1)

    # Drawdown
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=drawdown,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(255, 71, 87, 0.3)',
        line=dict(color=COLORS['loss'], width=1),
    ), row=2, col=1)

    # Mark maximum drawdown point
    max_dd_idx = drawdown.idxmin()
    fig.add_trace(go.Scatter(
        x=[df.loc[max_dd_idx, 'timestamp']],
        y=[drawdown[max_dd_idx]],
        mode='markers',
        name='Max Drawdown',
        marker=dict(color=COLORS['loss'], size=12, symbol='x'),
    ), row=2, col=1)

    apply_dark_theme(
        fig,
        title='Drawdown Analysis',
        showlegend=True,
    )

    return fig


def create_var_chart(df: pd.DataFrame) -> go.Figure:
    """Create Value at Risk (VaR) chart."""
    daily_pnl = df.groupby('date')['profit'].sum()

    if len(daily_pnl) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Not enough daily data for VaR analysis",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    var_95 = daily_pnl.quantile(0.05)
    var_99 = daily_pnl.quantile(0.01)

    fig = go.Figure()

    # Histogram of daily P&L
    fig.add_trace(go.Histogram(
        x=daily_pnl,
        nbinsx=30,
        marker_color=COLORS['neutral'],
        opacity=0.7,
        name='Daily P&L',
    ))

    # VaR lines
    fig.add_vline(
        x=var_95,
        line_dash='dash',
        line_color=COLORS['accent1'],
        annotation_text=f'95% VaR: ${var_95:.2f}',
        annotation_position='top left',
    )

    fig.add_vline(
        x=var_99,
        line_dash='dash',
        line_color=COLORS['loss'],
        annotation_text=f'99% VaR: ${var_99:.2f}',
        annotation_position='top left',
    )

    # Zero line
    fig.add_vline(x=0, line_dash='dot', line_color=COLORS['muted'])

    apply_dark_theme(
        fig,
        title='Value at Risk (VaR) Analysis',
        xaxis_title='Daily P&L (USDT)',
        yaxis_title='Frequency',
    )

    return fig


def create_streak_chart(stats: Dict) -> go.Figure:
    """Create streak analysis chart."""
    streaks = stats.get('streaks', {})

    win_streaks = streaks.get('win_streaks', [])
    loss_streaks = streaks.get('loss_streaks', [])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Streak Length Distribution', 'Streak Statistics')
    )

    # Streak distribution
    if win_streaks:
        fig.add_trace(go.Histogram(
            x=win_streaks,
            name='Win Streaks',
            marker_color=COLORS['profit'],
            opacity=0.7,
        ), row=1, col=1)

    if loss_streaks:
        fig.add_trace(go.Histogram(
            x=loss_streaks,
            name='Loss Streaks',
            marker_color=COLORS['loss'],
            opacity=0.7,
        ), row=1, col=1)

    # Streak statistics
    categories = ['Max Win', 'Max Loss', 'Avg Win', 'Avg Loss']
    values = [
        streaks.get('max_win_streak', 0),
        streaks.get('max_loss_streak', 0),
        streaks.get('avg_win_streak', 0),
        streaks.get('avg_loss_streak', 0),
    ]
    colors = [COLORS['profit'], COLORS['loss'], COLORS['profit'], COLORS['loss']]

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}' for v in values],
        textposition='outside',
    ), row=1, col=2)

    apply_dark_theme(
        fig,
        title='Streak Analysis',
        barmode='overlay',
    )

    return fig


def create_bet_after_result_chart(df: pd.DataFrame, stats: Dict) -> go.Figure:
    """Create chart comparing bet sizes after wins vs losses."""
    behavior = stats.get('behavior', {})

    avg_after_win = behavior.get('avg_bet_after_win', 0)
    avg_after_loss = behavior.get('avg_bet_after_loss', 0)
    chase_ratio = behavior.get('loss_chase_ratio', 1.0)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Bet After Result', 'Bet Size Distribution After Result')
    )

    # Bar chart comparison
    fig.add_trace(go.Bar(
        x=['After Win', 'After Loss'],
        y=[avg_after_win, avg_after_loss],
        marker_color=[COLORS['profit'], COLORS['loss']],
        text=[f'${avg_after_win:.2f}', f'${avg_after_loss:.2f}'],
        textposition='outside',
    ), row=1, col=1)

    # Distribution comparison
    if len(df) > 1:
        df_copy = df.copy()
        df_copy['prev_win'] = df_copy['is_win'].shift(1)
        df_copy = df_copy.dropna(subset=['prev_win'])

        after_win = df_copy[df_copy['prev_win'] == True]['bet_amount']
        after_loss = df_copy[df_copy['prev_win'] == False]['bet_amount']

        if len(after_win) > 0:
            fig.add_trace(go.Violin(
                y=after_win,
                name='After Win',
                marker_color=COLORS['profit'],
                box_visible=True,
                meanline_visible=True,
                side='negative',
            ), row=1, col=2)

        if len(after_loss) > 0:
            fig.add_trace(go.Violin(
                y=after_loss,
                name='After Loss',
                marker_color=COLORS['loss'],
                box_visible=True,
                meanline_visible=True,
                side='positive',
            ), row=1, col=2)

    # Add annotation about chasing
    chase_text = (
        f"Loss Chase Ratio: {chase_ratio:.2f}x\n"
        f"{'⚠️ Chasing losses detected!' if chase_ratio > 1.2 else '✓ Disciplined betting'}"
    )

    fig.add_annotation(
        text=chase_text,
        xref='paper', yref='paper',
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=12, color=COLORS['text']),
    )

    apply_dark_theme(
        fig,
        title='Betting Behavior: After Win vs After Loss',
        showlegend=False,
    )

    return fig


# =============================================================================
# INSIGHT GENERATION
# =============================================================================

def generate_insights(df: pd.DataFrame, stats: Dict) -> List[str]:
    """Generate text insights from the data."""
    insights = []

    # Best performing game
    if 'best_game' in stats:
        rtp = stats['game_stats'][stats['best_game']].get('rtp', 0)
        insights.append(
            f"Your best performing game is {stats['best_game'].title()} with "
            f"+${stats['best_game_profit']:.2f} profit ({rtp:.2f}% RTP)"
        )

    # Betting time pattern
    behavior = stats.get('behavior', {})
    if 'night_betting_pct' in behavior:
        pct = behavior['night_betting_pct']
        if pct > 50:
            insights.append(
                f"You place {pct:.1f}% of your bets between 10 PM and 4 AM"
            )

    # Loss chasing detection
    if 'loss_chase_ratio' in behavior:
        ratio = behavior['loss_chase_ratio']
        if ratio > 1.2:
            increase_pct = (ratio - 1) * 100
            insights.append(
                f"⚠️ Your average bet size increases by {increase_pct:.0f}% after a loss - "
                f"consider monitoring for loss chasing behavior"
            )

    # Lucky day
    daily_stats = stats.get('daily_stats', {})
    if daily_stats and 'mean' in daily_stats:
        best_day = max(daily_stats['mean'].items(), key=lambda x: x[1])
        if best_day[1] > 0:
            insights.append(
                f"{best_day[0]} is your luckiest day with average daily profit of +${best_day[1]:.2f}"
            )

    # Big wins count
    if stats.get('max_multiplier', 0) >= 100:
        big_wins = df[(df['multiplier'] >= 100) & (df['is_win'])]
        total_big = big_wins['payout_amount'].sum()
        insights.append(
            f"You've hit 100x+ multiplier {len(big_wins)} times, totaling ${total_big:.2f} in winnings"
        )

    # Streak info
    streaks = stats.get('streaks', {})
    if streaks.get('max_loss_streak', 0) >= 10:
        insights.append(
            f"Your longest losing streak was {streaks['max_loss_streak']} consecutive bets"
        )

    # Current drawdown warning
    if stats.get('max_drawdown', 0) < -50:
        insights.append(
            f"⚠️ Maximum drawdown reached ${abs(stats['max_drawdown']):.2f} "
            f"({abs(stats['max_drawdown_pct']):.1f}% from peak)"
        )

    # RTP insight
    if stats.get('rtp', 0) > 100:
        insights.append(
            f"Overall RTP of {stats['rtp']:.2f}% - you're beating the house! 🎉"
        )
    elif stats.get('rtp', 0) < 90:
        insights.append(
            f"Overall RTP of {stats['rtp']:.2f}% - below typical casino averages (95-97%)"
        )

    return insights


# =============================================================================
# HTML DASHBOARD GENERATION
# =============================================================================

def generate_html_dashboard(
    figures: Dict[str, go.Figure],
    stats: Dict,
    insights: List[str],
    output_path: str = 'betting_dashboard.html'
) -> str:
    """
    Generate a complete HTML dashboard with all visualizations.

    Args:
        figures: Dictionary of chart name to Plotly figure
        stats: Statistics dictionary
        insights: List of insight strings
        output_path: Path to save the HTML file

    Returns:
        Path to the generated HTML file
    """

    # Generate stats cards HTML
    stats_cards = f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Total Wagered</div>
            <div class="stat-value">${stats.get('total_wagered', 0):,.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Payouts</div>
            <div class="stat-value">${stats.get('total_payouts', 0):,.2f}</div>
        </div>
        <div class="stat-card {'profit' if stats.get('net_profit', 0) >= 0 else 'loss'}">
            <div class="stat-label">Net Profit</div>
            <div class="stat-value">${stats.get('net_profit', 0):+,.2f}</div>
            <div class="stat-sub">({stats.get('profit_percentage', 0):+.2f}%)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Win Rate</div>
            <div class="stat-value">{stats.get('win_rate', 0):.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Bets</div>
            <div class="stat-value">{stats.get('total_bets', 0):,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Best Game</div>
            <div class="stat-value">{stats.get('best_game', 'N/A').title()}</div>
            <div class="stat-sub profit">+${stats.get('best_game_profit', 0):.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Biggest Win</div>
            <div class="stat-value profit">${stats.get('biggest_win', 0):.2f}</div>
            <div class="stat-sub">({stats.get('biggest_win_multiplier', 0):.1f}x)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Max Drawdown</div>
            <div class="stat-value loss">${stats.get('max_drawdown', 0):.2f}</div>
            <div class="stat-sub">({stats.get('max_drawdown_pct', 0):.1f}%)</div>
        </div>
    </div>
    """

    # Generate insights HTML
    insights_html = ""
    if insights:
        insights_items = "\n".join([f'<li>{insight}</li>' for insight in insights])
        insights_html = f"""
        <div class="insights-section">
            <h2>📊 Key Insights</h2>
            <ul class="insights-list">
                {insights_items}
            </ul>
        </div>
        """

    # Generate chart HTML
    charts_html = ""
    chart_sections = {
        'Financial Overview': [
            'cumulative_pnl', 'pnl_by_game', 'wagered_vs_payout', 'daily_heatmap'
        ],
        'Betting Behavior': [
            'bet_distribution', 'betting_frequency', 'session_analysis',
            'hourly_pattern', 'day_of_week'
        ],
        'Multiplier & Wins': [
            'multiplier_distribution', 'big_wins_timeline', 'win_rate_trend',
            'risk_reward_scatter', 'multiplier_bet_heatmap'
        ],
        'Game Analysis': [
            'plinko_analysis', 'keno_analysis', 'game_comparison_radar'
        ],
        'Risk & Bankroll': [
            'drawdown', 'var_analysis', 'streak_analysis', 'bet_after_result'
        ],
    }

    nav_links = []
    for section_name, chart_keys in chart_sections.items():
        section_id = section_name.lower().replace(' ', '-').replace('&', 'and')
        nav_links.append(f'<a href="#{section_id}">{section_name}</a>')

        section_charts = ""
        for key in chart_keys:
            if key in figures and figures[key] is not None:
                chart_html = figures[key].to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    div_id=f'chart-{key}',
                    config={'displayModeBar': True, 'responsive': True}
                )
                section_charts += f'<div class="chart-container">{chart_html}</div>\n'

        if section_charts:
            charts_html += f"""
            <section id="{section_id}">
                <h2>{section_name}</h2>
                <div class="charts-grid">
                    {section_charts}
                </div>
            </section>
            """

    nav_html = "\n".join(nav_links)

    # Full HTML template
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Betting Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            line-height: 1.6;
        }}

        .dashboard {{
            display: flex;
            min-height: 100vh;
        }}

        .sidebar {{
            width: 220px;
            background-color: {COLORS['card_bg']};
            padding: 20px;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
        }}

        .sidebar h1 {{
            font-size: 1.2rem;
            margin-bottom: 20px;
            color: {COLORS['profit']};
        }}

        .sidebar nav {{
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        .sidebar a {{
            color: {COLORS['muted']};
            text-decoration: none;
            padding: 10px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }}

        .sidebar a:hover {{
            background-color: rgba(255, 255, 255, 0.1);
            color: {COLORS['text']};
        }}

        .main-content {{
            flex: 1;
            margin-left: 220px;
            padding: 30px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}

        .stat-card {{
            background-color: {COLORS['card_bg']};
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}

        .stat-card.profit .stat-value {{
            color: {COLORS['profit']};
        }}

        .stat-card.loss .stat-value {{
            color: {COLORS['loss']};
        }}

        .stat-label {{
            color: {COLORS['muted']};
            font-size: 0.9rem;
            margin-bottom: 8px;
        }}

        .stat-value {{
            font-size: 1.8rem;
            font-weight: bold;
        }}

        .stat-sub {{
            font-size: 0.85rem;
            color: {COLORS['muted']};
            margin-top: 5px;
        }}

        .stat-sub.profit {{
            color: {COLORS['profit']};
        }}

        .stat-sub.loss {{
            color: {COLORS['loss']};
        }}

        .insights-section {{
            background-color: {COLORS['card_bg']};
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 40px;
        }}

        .insights-section h2 {{
            margin-bottom: 15px;
            color: {COLORS['accent1']};
        }}

        .insights-list {{
            list-style: none;
        }}

        .insights-list li {{
            padding: 10px 0;
            border-bottom: 1px solid {COLORS['grid']};
        }}

        .insights-list li:last-child {{
            border-bottom: none;
        }}

        section {{
            margin-bottom: 50px;
        }}

        section h2 {{
            margin-bottom: 25px;
            padding-bottom: 10px;
            border-bottom: 2px solid {COLORS['grid']};
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 25px;
        }}

        .chart-container {{
            background-color: {COLORS['card_bg']};
            border-radius: 12px;
            padding: 15px;
            min-height: 450px;
        }}

        .chart-container > div {{
            width: 100% !important;
            height: 100% !important;
        }}

        @media (max-width: 900px) {{
            .sidebar {{
                display: none;
            }}

            .main-content {{
                margin-left: 0;
            }}

            .charts-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <aside class="sidebar">
            <h1>🎰 Betting Analysis</h1>
            <nav>
                <a href="#overview">Overview</a>
                {nav_html}
            </nav>
        </aside>

        <main class="main-content">
            <section id="overview">
                <h2>Dashboard Overview</h2>
                {stats_cards}
                {insights_html}
            </section>

            {charts_html}
        </main>
    </div>
</body>
</html>
    """

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"Dashboard saved to: {output_path}")
    return output_path


# =============================================================================
# PNG EXPORT
# =============================================================================

def save_charts_to_png(figures: Dict[str, go.Figure], output_dir: str = 'charts') -> List[str]:
    """
    Save all charts as high-resolution PNG files.

    Args:
        figures: Dictionary of chart name to Plotly figure
        output_dir: Directory to save PNG files

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # Map chart names to numbered filenames
    chart_order = [
        ('cumulative_pnl', '01_cumulative_pnl'),
        ('pnl_by_game', '02_pnl_by_game'),
        ('wagered_vs_payout', '03_wagered_vs_payout'),
        ('daily_heatmap', '04_daily_heatmap'),
        ('bet_distribution', '05_bet_distribution'),
        ('betting_frequency', '06_betting_frequency'),
        ('session_analysis', '07_session_analysis'),
        ('hourly_pattern', '08_hourly_pattern'),
        ('day_of_week', '09_day_of_week'),
        ('multiplier_distribution', '10_multiplier_distribution'),
        ('big_wins_timeline', '11_big_wins_timeline'),
        ('win_rate_trend', '12_win_rate_trend'),
        ('risk_reward_scatter', '13_risk_reward_scatter'),
        ('multiplier_bet_heatmap', '14_multiplier_bet_heatmap'),
        ('plinko_analysis', '15_plinko_analysis'),
        ('keno_analysis', '16_keno_analysis'),
        ('game_comparison_radar', '17_game_comparison_radar'),
        ('drawdown', '18_drawdown'),
        ('var_analysis', '19_var_analysis'),
        ('streak_analysis', '20_streak_analysis'),
        ('bet_after_result', '21_bet_after_result'),
    ]

    for key, filename in chart_order:
        if key in figures and figures[key] is not None:
            filepath = os.path.join(output_dir, f'{filename}.png')
            try:
                figures[key].write_image(
                    filepath,
                    width=1200,
                    height=700,
                    scale=2,
                )
                saved_files.append(filepath)
                print(f"Saved: {filepath}")
            except Exception as e:
                print(f"Warning: Could not save {filename}.png: {e}")

    return saved_files


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point for the betting visualizer."""
    parser = argparse.ArgumentParser(
        description='Betting Data Analysis & Visualization Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python betting_visualizer.py bet_archive.json
  python betting_visualizer.py nov_bets.json dec_bets.json
  python betting_visualizer.py data.json -o dashboard.html --export-png
  python betting_visualizer.py data.json --from 2025-11-01 --to 2025-11-30
        """
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='JSON file(s) containing betting data'
    )
    parser.add_argument(
        '-o', '--output',
        default='betting_dashboard.html',
        help='Output HTML file path (default: betting_dashboard.html)'
    )
    parser.add_argument(
        '--export-png',
        action='store_true',
        help='Export charts as PNG files to ./charts folder'
    )
    parser.add_argument(
        '--png-dir',
        default='charts',
        help='Directory for PNG exports (default: charts)'
    )
    parser.add_argument(
        '--from',
        dest='date_from',
        help='Filter bets from this date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--to',
        dest='date_to',
        help='Filter bets until this date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--big-win-threshold',
        type=float,
        default=5.0,
        help='Multiplier threshold for "big win" charts (default: 5.0)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BETTING DATA ANALYSIS & VISUALIZATION")
    print("=" * 60)

    # Load data
    print("\n📂 Loading data...")
    try:
        df = load_data(args.files)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Preprocess data
    print("\n🔧 Preprocessing data...")
    df = preprocess_data(df)

    # Apply date filters
    if args.date_from:
        try:
            date_from = pd.to_datetime(args.date_from)
            df = df[df['timestamp'] >= date_from]
            print(f"  Filtered from: {args.date_from}")
        except Exception as e:
            print(f"Warning: Invalid from date: {e}")

    if args.date_to:
        try:
            date_to = pd.to_datetime(args.date_to)
            df = df[df['timestamp'] <= date_to]
            print(f"  Filtered to: {args.date_to}")
        except Exception as e:
            print(f"Warning: Invalid to date: {e}")

    if len(df) == 0:
        print("Error: No bets found after filtering")
        sys.exit(1)

    print(f"  Total bets after filtering: {len(df)}")

    # Calculate statistics
    print("\n📊 Calculating statistics...")
    stats = calculate_statistics(df)

    # Generate insights
    print("\n💡 Generating insights...")
    insights = generate_insights(df, stats)
    for insight in insights:
        print(f"  • {insight}")

    # Create visualizations
    print("\n📈 Creating visualizations...")
    figures = {}

    # Financial Overview
    print("  Creating financial overview charts...")
    figures['cumulative_pnl'] = create_cumulative_pnl_chart(df)
    figures['pnl_by_game'] = create_pnl_by_game_chart(df, stats)
    figures['wagered_vs_payout'] = create_wagered_vs_payout_chart(df)
    figures['daily_heatmap'] = create_daily_heatmap(df)

    # Betting Behavior
    print("  Creating betting behavior charts...")
    figures['bet_distribution'] = create_bet_distribution_chart(df)
    figures['betting_frequency'] = create_betting_frequency_chart(df)
    figures['session_analysis'] = create_session_analysis_chart(df)
    figures['hourly_pattern'] = create_hourly_pattern_chart(df)
    figures['day_of_week'] = create_day_of_week_chart(df)

    # Multiplier & Wins
    print("  Creating multiplier & wins charts...")
    figures['multiplier_distribution'] = create_multiplier_distribution_chart(df)
    figures['big_wins_timeline'] = create_big_wins_timeline_chart(df, args.big_win_threshold)
    figures['win_rate_trend'] = create_win_rate_trend_chart(df)
    figures['risk_reward_scatter'] = create_risk_reward_scatter(df)
    figures['multiplier_bet_heatmap'] = create_multiplier_bet_heatmap(df)

    # Game-Specific
    print("  Creating game-specific charts...")
    figures['plinko_analysis'] = create_plinko_analysis(df)
    figures['keno_analysis'] = create_keno_analysis(df)
    figures['game_comparison_radar'] = create_game_comparison_radar(df, stats)

    # Risk & Bankroll
    print("  Creating risk analysis charts...")
    figures['drawdown'] = create_drawdown_chart(df)
    figures['var_analysis'] = create_var_chart(df)
    figures['streak_analysis'] = create_streak_chart(stats)
    figures['bet_after_result'] = create_bet_after_result_chart(df, stats)

    # Generate HTML dashboard
    print("\n🌐 Generating HTML dashboard...")
    html_path = generate_html_dashboard(figures, stats, insights, args.output)

    # Export PNGs if requested
    if args.export_png:
        print("\n🖼️ Exporting PNG charts...")
        try:
            saved_files = save_charts_to_png(figures, args.png_dir)
            print(f"  Exported {len(saved_files)} charts to {args.png_dir}/")
        except Exception as e:
            print(f"  Warning: PNG export failed: {e}")
            print("  Install kaleido for PNG export: pip install kaleido")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total Bets: {stats['total_bets']:,}")
    print(f"  Total Wagered: ${stats['total_wagered']:,.2f}")
    print(f"  Net Profit: ${stats['net_profit']:+,.2f} ({stats['profit_percentage']:+.2f}%)")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    print(f"  RTP: {stats['rtp']:.2f}%")
    print(f"\n  Dashboard: {html_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
