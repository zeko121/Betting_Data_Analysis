"""Game-specific analysis charts."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

from .theme import COLORS, GAME_COLORS, apply_dark_theme


def create_plinko_analysis(df: pd.DataFrame) -> go.Figure:
    """Create Plinko-specific analysis charts."""
    plinko = df[df['game'] == 'plinko']

    if len(plinko) == 0 or 'plinko_risk' not in plinko.columns:
        fig = go.Figure()
        fig.add_annotation(text="No Plinko data available",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Plinko Deep Dive Analysis')

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
        return apply_dark_theme(fig, title='Keno Deep Dive Analysis')

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

    def parse_numbers(val):
        """Parse comma-separated string or list into list of integers."""
        if isinstance(val, list):
            return val
        if isinstance(val, str) and val:
            try:
                return [int(x.strip()) for x in val.split(',') if x.strip()]
            except ValueError:
                return []
        return []

    for _, row in keno.iterrows():
        selected = parse_numbers(row.get('keno_selected', ''))
        drawn = parse_numbers(row.get('keno_drawn', ''))

        for num in selected:
            selected_counts[num] = selected_counts.get(num, 0) + 1

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


def create_game_comparison_radar(df: pd.DataFrame, stats: Dict[str, Any]) -> go.Figure:
    """Create radar chart comparing all games across multiple metrics."""
    game_stats = stats.get('game_stats', {})

    if not game_stats:
        fig = go.Figure()
        fig.add_annotation(text="No game statistics available",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Game Comparison Radar')

    categories = ['Win Rate', 'Avg Multiplier', 'RTP', 'Bet Frequency', 'Avg Bet']

    fig = go.Figure()

    # Get max values for normalization
    max_bets = max(gs.get('num_bets', 1) for gs in game_stats.values())
    max_bet = max(gs.get('avg_bet', 1) for gs in game_stats.values())

    for game, gs in game_stats.items():
        # Normalize values to 0-100 scale for radar
        values = [
            gs.get('win_rate', 0) * 100,  # Already 0-1
            min(gs.get('avg_mult', 0) * 10, 100),  # Scale multiplier
            min(gs.get('rtp', 0), 150),  # RTP can exceed 100
            (gs.get('num_bets', 0) / max_bets * 100) if max_bets > 0 else 0,
            (gs.get('avg_bet', 0) / max_bet * 100) if max_bet > 0 else 0,
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


def create_game_breakdown_table(df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
    """Create a summary table of game statistics."""
    game_stats = stats.get('game_stats', {})

    if not game_stats:
        return pd.DataFrame()

    rows = []
    for game, gs in game_stats.items():
        rows.append({
            'Game': game.title(),
            'Bets': gs.get('num_bets', 0),
            'Wagered': f"${gs.get('total_wagered', 0):,.2f}",
            'Payouts': f"${gs.get('total_payouts', 0):,.2f}",
            'Profit': f"${gs.get('net_profit', 0):+,.2f}",
            'Win Rate': f"{gs.get('win_rate', 0) * 100:.1f}%",
            'RTP': f"{gs.get('rtp', 0):.1f}%",
            'Avg Bet': f"${gs.get('avg_bet', 0):.2f}",
        })

    return pd.DataFrame(rows)
