"""Theme configuration for charts."""

import plotly.graph_objects as go

# Color palette
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

# Game-specific colors
GAME_COLORS = {
    'plinko': '#00d4aa',
    'keno': '#ff9ff3',
    'flip': '#feca57',
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

# Base Plotly layout
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
    """
    Apply dark theme styling to a Plotly figure.

    Args:
        fig: Plotly figure to style
        title: Optional title for the chart
        **kwargs: Additional layout parameters

    Returns:
        Styled figure
    """
    layout_update = {**PLOTLY_LAYOUT, **kwargs}
    if title:
        layout_update['title'] = {
            'text': title,
            'font': {'size': 18, 'color': COLORS['text']}
        }
    fig.update_layout(**layout_update)
    return fig


def get_game_color(game_name: str) -> str:
    """Get color for a specific game."""
    return GAME_COLORS.get(game_name.lower(), COLORS['muted'])
