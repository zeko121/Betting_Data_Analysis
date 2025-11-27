"""Financial overview charts."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

from .theme import COLORS, GAME_COLORS, apply_dark_theme


def create_cumulative_pnl_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create cumulative profit/loss over time chart.
    The most important chart - shows the overall journey.
    """
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Cumulative Profit/Loss Over Time')

    fig = go.Figure()

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


def create_pnl_by_game_chart(df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart showing P&L by game."""
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Profit/Loss by Game')

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
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Total Wagered vs Total Payouts by Game')

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
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Daily Profit/Loss Heatmap')

    daily_profit = df.groupby('date')['profit'].sum().reset_index()
    daily_profit['date'] = pd.to_datetime(daily_profit['date'])
    daily_profit['week'] = daily_profit['date'].dt.isocalendar().week
    daily_profit['day_of_week'] = daily_profit['date'].dt.dayofweek

    # Create pivot table for heatmap
    pivot = daily_profit.pivot_table(
        index='day_of_week',
        columns='week',
        values='profit',
        aggfunc='sum'
    )

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
