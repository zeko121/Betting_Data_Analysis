"""Betting behavior analysis charts."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

from .theme import COLORS, GAME_COLORS, apply_dark_theme


def create_bet_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create histogram showing bet size distribution."""
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Bet Size Distribution Analysis')

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
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Betting Frequency Over Time (by Game)')

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
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Session Analysis: Duration vs Profit')

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
    session_data = session_data[session_data['duration_mins'] > 0]

    if len(session_data) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No session data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Session Analysis: Duration vs Profit')

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
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Betting Patterns by Hour of Day')

    hourly = df.groupby('hour').agg({
        'profit': ['sum', 'mean'],
        'bet_amount': 'count',
    })
    hourly.columns = ['total_profit', 'avg_profit', 'num_bets']
    hourly = hourly.reindex(range(24), fill_value=0)

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
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Daily Profit/Loss Distribution by Day of Week')

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
