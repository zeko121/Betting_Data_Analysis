"""Risk and bankroll analysis charts."""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

from .theme import COLORS, GAME_COLORS, apply_dark_theme


def create_drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """Create maximum drawdown analysis chart."""
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Drawdown Analysis')

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
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Value at Risk (VaR) Analysis')

    daily_pnl = df.groupby('date')['profit'].sum()

    if len(daily_pnl) < 5:
        fig = go.Figure()
        fig.add_annotation(text="Not enough daily data for VaR analysis",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Value at Risk (VaR) Analysis')

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


def create_streak_chart(stats: Dict[str, Any]) -> go.Figure:
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


def create_bet_after_result_chart(df: pd.DataFrame, stats: Dict[str, Any]) -> go.Figure:
    """Create chart comparing bet sizes after wins vs losses."""
    if len(df) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Betting Behavior: After Win vs After Loss')

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
    if chase_ratio > 1.2:
        chase_text = f"Loss Chase Ratio: {chase_ratio:.2f}x - Chasing losses detected!"
    else:
        chase_text = f"Loss Chase Ratio: {chase_ratio:.2f}x - Disciplined betting"

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


def create_multiplier_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create histogram showing distribution of winning multipliers."""
    wins = df[df['is_win'] & (df['multiplier'] > 0)]

    if len(wins) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No winning bets to display",
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Winning Multiplier Distribution')

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
        return apply_dark_theme(fig, title=f'Big Wins Timeline (>={threshold_mult}x Multiplier)')

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
        title=f'Big Wins Timeline (>={threshold_mult}x Multiplier)',
        xaxis_title='Date',
        yaxis_title='Payout Amount (USDT)',
    )

    return fig


def create_win_rate_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Create rolling win rate chart over time."""
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return apply_dark_theme(fig, title='Rolling Win Rate Over Time')

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
