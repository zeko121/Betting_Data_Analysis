"""Auto-generated insight utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


def generate_insights(df: pd.DataFrame, stats: Dict[str, Any]) -> List[str]:
    """
    Generate text insights from the data.

    Args:
        df: Preprocessed betting DataFrame
        stats: Statistics dictionary from calculate_statistics

    Returns:
        List of insight strings
    """
    if len(df) == 0 or not stats:
        return ["No data available to generate insights."]

    insights = []

    # Best performing game
    if stats.get('best_game') and stats.get('best_game') != 'N/A':
        game_stats = stats.get('game_stats', {})
        rtp = game_stats.get(stats['best_game'], {}).get('rtp', 0)
        insights.append(
            f"Your best performing game is **{stats['best_game'].title()}** with "
            f"**+${stats['best_game_profit']:.2f}** profit ({rtp:.2f}% RTP)"
        )

    # Worst performing game
    if (stats.get('worst_game') and stats.get('worst_game') != 'N/A' and
        stats.get('worst_game_profit', 0) < 0):
        insights.append(
            f"Your worst performing game is **{stats['worst_game'].title()}** with "
            f"**${stats['worst_game_profit']:.2f}** loss"
        )

    # Betting time pattern
    behavior = stats.get('behavior', {})
    if behavior.get('night_betting_pct', 0) > 50:
        pct = behavior['night_betting_pct']
        insights.append(
            f"You place **{pct:.1f}%** of your bets between 10 PM and 4 AM"
        )

    # Peak hour
    if behavior.get('peak_hour') is not None:
        hour = behavior['peak_hour']
        pct = behavior.get('peak_hour_pct', 0)
        hour_str = f"{hour}:00" if hour < 12 else f"{hour}:00"
        am_pm = "AM" if hour < 12 else "PM"
        display_hour = hour if hour <= 12 else hour - 12
        if display_hour == 0:
            display_hour = 12
        insights.append(
            f"Your most active betting hour is **{display_hour}:00 {am_pm}** ({pct:.1f}% of bets)"
        )

    # Loss chasing detection
    if behavior.get('loss_chase_ratio', 1.0) > 1.2:
        ratio = behavior['loss_chase_ratio']
        increase_pct = (ratio - 1) * 100
        insights.append(
            f"Your average bet size increases by **{increase_pct:.0f}%** after a loss - "
            f"this may indicate loss chasing behavior"
        )
    elif behavior.get('loss_chase_ratio', 1.0) < 0.9:
        insights.append(
            "You show disciplined betting - your bet size decreases after losses"
        )

    # Lucky day
    daily_stats = stats.get('daily_stats', {})
    if daily_stats and 'mean' in daily_stats:
        means = daily_stats['mean']
        if means:
            best_day = max(means.items(), key=lambda x: x[1])
            if best_day[1] > 0:
                insights.append(
                    f"**{best_day[0]}** is your luckiest day with average daily profit of **+${best_day[1]:.2f}**"
                )

            worst_day = min(means.items(), key=lambda x: x[1])
            if worst_day[1] < 0:
                insights.append(
                    f"**{worst_day[0]}** is your worst day with average daily loss of **${worst_day[1]:.2f}**"
                )

    # Big wins count
    if stats.get('max_multiplier', 0) >= 10:
        big_wins = df[(df['multiplier'] >= 10) & (df['is_win'])]
        if len(big_wins) > 0:
            total_big = big_wins['payout_amount'].sum()
            insights.append(
                f"You've hit **10x+** multiplier **{len(big_wins)}** times, totaling **${total_big:.2f}** in winnings"
            )

    # Biggest single win
    if stats.get('biggest_win', 0) > 0:
        mult = stats.get('biggest_win_multiplier', 0)
        game = stats.get('biggest_win_game', 'unknown')
        insights.append(
            f"Your biggest single win was **${stats['biggest_win']:.2f}** ({mult:.1f}x) on **{game.title()}**"
        )

    # Streak info
    streaks = stats.get('streaks', {})
    if streaks.get('max_win_streak', 0) >= 5:
        insights.append(
            f"Your longest winning streak was **{streaks['max_win_streak']}** consecutive wins!"
        )

    if streaks.get('max_loss_streak', 0) >= 10:
        insights.append(
            f"Your longest losing streak was **{streaks['max_loss_streak']}** consecutive losses"
        )

    # Drawdown warning
    if stats.get('max_drawdown', 0) < -50:
        insights.append(
            f"Maximum drawdown reached **${abs(stats['max_drawdown']):.2f}** "
            f"({abs(stats['max_drawdown_pct']):.1f}% from peak)"
        )

    # RTP insight
    if stats.get('rtp', 0) > 100:
        insights.append(
            f"Overall RTP of **{stats['rtp']:.2f}%** - you're beating the house!"
        )
    elif stats.get('rtp', 0) < 90:
        insights.append(
            f"Overall RTP of **{stats['rtp']:.2f}%** - below typical casino averages (95-97%)"
        )

    # Session insight
    if stats.get('avg_session_duration', 0) > 60:
        insights.append(
            f"Your average session lasts **{stats['avg_session_duration']:.0f} minutes** - "
            f"consider setting time limits"
        )

    # Bet count milestone
    total_bets = stats.get('total_bets', 0)
    if total_bets >= 10000:
        insights.append(f"You've placed over **{total_bets:,}** bets!")
    elif total_bets >= 1000:
        insights.append(f"You've placed over **{total_bets:,}** bets")

    # Win rate context
    win_rate = stats.get('win_rate', 0)
    if win_rate > 55:
        insights.append(f"Your win rate of **{win_rate:.1f}%** is above average")
    elif win_rate < 40:
        insights.append(f"Your win rate of **{win_rate:.1f}%** is below average")

    # If no insights generated, add a default
    if not insights:
        insights.append("Upload more data to see personalized insights")

    return insights


def get_alert_insights(df: pd.DataFrame, stats: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate alert-style insights for concerning patterns.

    Returns list of dicts with 'type' (warning/info/success) and 'message'
    """
    alerts = []

    if not stats:
        return alerts

    behavior = stats.get('behavior', {})

    # Loss chasing alert
    if behavior.get('loss_chase_ratio', 1.0) > 1.5:
        alerts.append({
            'type': 'warning',
            'message': f"Loss chasing detected: bet size increases {((behavior['loss_chase_ratio']-1)*100):.0f}% after losses"
        })

    # Drawdown alert
    if stats.get('max_drawdown', 0) < -100:
        alerts.append({
            'type': 'warning',
            'message': f"Significant drawdown: ${abs(stats['max_drawdown']):.2f} from peak"
        })

    # Long losing streak alert
    streaks = stats.get('streaks', {})
    if streaks.get('max_loss_streak', 0) >= 15:
        alerts.append({
            'type': 'warning',
            'message': f"Long losing streak detected: {streaks['max_loss_streak']} consecutive losses"
        })

    # Night betting alert
    if behavior.get('night_betting_pct', 0) > 70:
        alerts.append({
            'type': 'info',
            'message': f"Late night betting: {behavior['night_betting_pct']:.0f}% of bets between 10PM-4AM"
        })

    # Positive alerts
    if stats.get('net_profit', 0) > 0:
        alerts.append({
            'type': 'success',
            'message': f"You're in profit: +${stats['net_profit']:.2f}"
        })

    if streaks.get('max_win_streak', 0) >= 10:
        alerts.append({
            'type': 'success',
            'message': f"Great winning streak: {streaks['max_win_streak']} consecutive wins!"
        })

    return alerts
