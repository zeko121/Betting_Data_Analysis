"""Chart modules for betting data visualization."""

from .financial import (
    create_cumulative_pnl_chart,
    create_pnl_by_game_chart,
    create_wagered_vs_payout_chart,
    create_daily_heatmap,
)
from .behavior import (
    create_bet_distribution_chart,
    create_betting_frequency_chart,
    create_session_analysis_chart,
    create_hourly_pattern_chart,
    create_day_of_week_chart,
)
from .games import (
    create_plinko_analysis,
    create_keno_analysis,
    create_game_comparison_radar,
)
from .risk import (
    create_drawdown_chart,
    create_var_chart,
    create_streak_chart,
    create_bet_after_result_chart,
    create_multiplier_distribution_chart,
    create_big_wins_timeline_chart,
    create_win_rate_trend_chart,
)
from .theme import COLORS, GAME_COLORS, apply_dark_theme

__all__ = [
    # Theme
    'COLORS',
    'GAME_COLORS',
    'apply_dark_theme',
    # Financial
    'create_cumulative_pnl_chart',
    'create_pnl_by_game_chart',
    'create_wagered_vs_payout_chart',
    'create_daily_heatmap',
    # Behavior
    'create_bet_distribution_chart',
    'create_betting_frequency_chart',
    'create_session_analysis_chart',
    'create_hourly_pattern_chart',
    'create_day_of_week_chart',
    # Games
    'create_plinko_analysis',
    'create_keno_analysis',
    'create_game_comparison_radar',
    # Risk
    'create_drawdown_chart',
    'create_var_chart',
    'create_streak_chart',
    'create_bet_after_result_chart',
    'create_multiplier_distribution_chart',
    'create_big_wins_timeline_chart',
    'create_win_rate_trend_chart',
]
