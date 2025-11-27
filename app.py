"""
Betting Analytics Dashboard - Streamlit Web App

A comprehensive interactive dashboard for analyzing crypto casino betting history.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta

# Import utilities
from utils.data_loader import load_json_data, preprocess_data, apply_filters
from utils.calculations import calculate_statistics, get_session_data
from utils.insights import generate_insights, get_alert_insights

# Import charts
from charts.financial import (
    create_cumulative_pnl_chart,
    create_pnl_by_game_chart,
    create_wagered_vs_payout_chart,
    create_daily_heatmap,
)
from charts.behavior import (
    create_bet_distribution_chart,
    create_betting_frequency_chart,
    create_session_analysis_chart,
    create_hourly_pattern_chart,
    create_day_of_week_chart,
)
from charts.games import (
    create_plinko_analysis,
    create_keno_analysis,
    create_game_comparison_radar,
    create_game_breakdown_table,
)
from charts.risk import (
    create_drawdown_chart,
    create_var_chart,
    create_streak_chart,
    create_bet_after_result_chart,
    create_multiplier_distribution_chart,
    create_big_wins_timeline_chart,
    create_win_rate_trend_chart,
)
from charts.theme import COLORS

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Betting Analytics",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #0f0f1a;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 170, 0.2);
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-label {
        color: #888888;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #ffffff;
    }

    .metric-value.profit {
        color: #00d4aa;
    }

    .metric-value.loss {
        color: #ff4757;
    }

    .metric-delta {
        font-size: 0.85rem;
        color: #888888;
    }

    /* Insight boxes */
    .insight-box {
        background: #16213e;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #00d4aa;
    }

    .insight-warning {
        border-left-color: #ff4757;
    }

    .insight-info {
        border-left-color: #5dade2;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 10px 20px;
        color: #ffffff;
    }

    .stTabs [aria-selected="true"] {
        background-color: #00d4aa;
        color: #0f0f1a;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #16213e;
        border-radius: 8px;
        padding: 1rem;
    }

    /* Download button */
    .stDownloadButton > button {
        background-color: #00d4aa;
        color: #0f0f1a;
        border: none;
        border-radius: 8px;
    }

    .stDownloadButton > button:hover {
        background-color: #00b894;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def render_metric_card(label: str, value: str, delta: str = None, value_class: str = ""):
    """Render a styled metric card."""
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {value_class}">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def format_currency(value: float) -> str:
    """Format value as currency."""
    if value >= 0:
        return f"${value:,.2f}"
    else:
        return f"-${abs(value):,.2f}"


def format_profit(value: float) -> str:
    """Format value as profit/loss."""
    return f"${value:+,.2f}"


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🎰 Betting Analytics")
    st.markdown("---")

    # File upload
    st.subheader("📁 Upload Data")
    uploaded_files = st.file_uploader(
        "Drop your bet archive JSON files here",
        type=['json'],
        accept_multiple_files=True,
        help="Upload one or more JSON files from your betting history export"
    )

    if uploaded_files:
        try:
            # Combine all uploaded files
            all_dfs = []
            for file in uploaded_files:
                raw_df = load_json_data(file.read())
                all_dfs.append(raw_df)

            combined_raw = pd.concat(all_dfs, ignore_index=True)
            st.session_state.raw_df = preprocess_data(combined_raw)

            # Store file info
            total_bets = len(st.session_state.raw_df)
            date_range = (
                st.session_state.raw_df['timestamp'].min(),
                st.session_state.raw_df['timestamp'].max()
            )
            st.session_state.file_info = {
                'files': len(uploaded_files),
                'total_bets': total_bets,
                'date_range': date_range,
            }

            st.success(f"✅ Loaded {total_bets:,} bets from {len(uploaded_files)} file(s)")
            st.caption(f"Date range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")

        except Exception as e:
            st.error(f"Error loading files: {str(e)}")
            st.session_state.raw_df = None

    # Filters (only show if data is loaded)
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        st.markdown("---")
        st.subheader("🔍 Filters")

        # Date range filter
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()

        date_range = st.date_input(
            "📅 Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # Game filter
        all_games = sorted(df['game'].unique().tolist())
        selected_games = st.multiselect(
            "🎮 Games",
            options=all_games,
            default=all_games,
        )

        # Bet size filter
        min_bet = float(df['bet_amount'].min())
        max_bet = float(df['bet_amount'].max())

        if max_bet > min_bet:
            bet_range = st.slider(
                "💰 Bet Size Range",
                min_value=min_bet,
                max_value=max_bet,
                value=(min_bet, max_bet),
                format="$%.2f",
            )
        else:
            bet_range = (min_bet, max_bet)

        # Result filter
        result_filter = st.radio(
            "🎯 Show",
            options=["All Bets", "Wins Only", "Losses Only"],
            horizontal=True,
        )

        result_map = {
            "All Bets": "all",
            "Wins Only": "wins",
            "Losses Only": "losses",
        }

        # Apply filters
        filtered_df = apply_filters(
            df,
            date_range=date_range,
            selected_games=selected_games,
            bet_range=bet_range,
            result_filter=result_map[result_filter],
        )

        st.session_state.df = filtered_df
        st.session_state.stats = calculate_statistics(filtered_df)

        # Filter summary
        st.markdown("---")
        st.caption(f"Showing {len(filtered_df):,} of {len(df):,} bets")

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data",
            csv,
            "betting_data.csv",
            "text/csv",
            use_container_width=True,
        )


# =============================================================================
# MAIN CONTENT
# =============================================================================

if st.session_state.df is not None and len(st.session_state.df) > 0:
    df = st.session_state.df
    stats = st.session_state.stats

    # =========================================================================
    # SUMMARY METRICS ROW
    # =========================================================================

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        render_metric_card(
            "Total Wagered",
            format_currency(stats.get('total_wagered', 0)),
        )

    with col2:
        render_metric_card(
            "Total Payouts",
            format_currency(stats.get('total_payouts', 0)),
        )

    with col3:
        net_profit = stats.get('net_profit', 0)
        profit_pct = stats.get('profit_percentage', 0)
        render_metric_card(
            "Net Profit",
            format_profit(net_profit),
            f"({profit_pct:+.2f}%)",
            "profit" if net_profit >= 0 else "loss",
        )

    with col4:
        render_metric_card(
            "Win Rate",
            f"{stats.get('win_rate', 0):.1f}%",
        )

    with col5:
        render_metric_card(
            "Total Bets",
            f"{stats.get('total_bets', 0):,}",
        )

    st.markdown("---")

    # =========================================================================
    # TABS
    # =========================================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🎮 Games",
        "⏰ Behavior",
        "⚠️ Risk",
        "📋 Data"
    ])

    # -------------------------------------------------------------------------
    # OVERVIEW TAB
    # -------------------------------------------------------------------------

    with tab1:
        # Cumulative P&L Chart (full width)
        st.subheader("Cumulative Profit/Loss")
        fig_pnl = create_cumulative_pnl_chart(df)
        st.plotly_chart(fig_pnl, use_container_width=True)

        # Two column layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("P&L by Game")
            fig_game = create_pnl_by_game_chart(df)
            st.plotly_chart(fig_game, use_container_width=True)

        with col2:
            st.subheader("Wagered vs Payouts")
            fig_vs = create_wagered_vs_payout_chart(df)
            st.plotly_chart(fig_vs, use_container_width=True)

        # Daily heatmap
        st.subheader("Daily Performance Heatmap")
        fig_heat = create_daily_heatmap(df)
        st.plotly_chart(fig_heat, use_container_width=True)

        # Insights Section
        st.markdown("---")
        st.subheader("💡 Auto-Generated Insights")

        insights = generate_insights(df, stats)
        alerts = get_alert_insights(df, stats)

        # Show alerts first
        for alert in alerts:
            alert_type = alert.get('type', 'info')
            if alert_type == 'warning':
                st.warning(alert['message'])
            elif alert_type == 'success':
                st.success(alert['message'])
            else:
                st.info(alert['message'])

        # Show insights in expander
        with st.expander("View All Insights", expanded=True):
            for insight in insights:
                st.markdown(f"• {insight}")

    # -------------------------------------------------------------------------
    # GAMES TAB
    # -------------------------------------------------------------------------

    with tab2:
        st.subheader("Game Performance Comparison")

        # Game comparison radar
        fig_radar = create_game_comparison_radar(df, stats)
        st.plotly_chart(fig_radar, use_container_width=True)

        # Game breakdown table
        st.subheader("Game Statistics Table")
        game_table = create_game_breakdown_table(df, stats)
        if len(game_table) > 0:
            st.dataframe(
                game_table,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")

        # Game-specific deep dives
        st.subheader("Game Deep Dive")
        available_games = df['game'].unique().tolist()

        if 'plinko' in available_games:
            with st.expander("🎯 Plinko Analysis", expanded=False):
                fig_plinko = create_plinko_analysis(df)
                st.plotly_chart(fig_plinko, use_container_width=True)

        if 'keno' in available_games:
            with st.expander("🎱 Keno Analysis", expanded=False):
                fig_keno = create_keno_analysis(df)
                st.plotly_chart(fig_keno, use_container_width=True)

    # -------------------------------------------------------------------------
    # BEHAVIOR TAB
    # -------------------------------------------------------------------------

    with tab3:
        st.subheader("Betting Patterns")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Bet Size Distribution")
            fig_dist = create_bet_distribution_chart(df)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            st.markdown("##### Hourly Patterns")
            fig_hourly = create_hourly_pattern_chart(df)
            st.plotly_chart(fig_hourly, use_container_width=True)

        st.markdown("---")

        st.subheader("Time Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Betting Frequency Over Time")
            fig_freq = create_betting_frequency_chart(df)
            st.plotly_chart(fig_freq, use_container_width=True)

        with col2:
            st.markdown("##### Day of Week Performance")
            fig_dow = create_day_of_week_chart(df)
            st.plotly_chart(fig_dow, use_container_width=True)

        st.markdown("---")

        st.subheader("Session Analysis")
        fig_session = create_session_analysis_chart(df)
        st.plotly_chart(fig_session, use_container_width=True)

        # Session summary
        session_data = get_session_data(df)
        if len(session_data) > 0:
            with st.expander("Session Details"):
                st.caption(f"Total sessions: {len(session_data)}")
                st.caption(f"Average session duration: {stats.get('avg_session_duration', 0):.0f} minutes")
                st.caption(f"Average bets per session: {stats.get('avg_session_bets', 0):.0f}")

    # -------------------------------------------------------------------------
    # RISK TAB
    # -------------------------------------------------------------------------

    with tab4:
        st.subheader("Risk Metrics")

        # Quick metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            max_dd = stats.get('max_drawdown', 0)
            st.metric("Max Drawdown", format_currency(abs(max_dd)))

        with col2:
            max_dd_pct = stats.get('max_drawdown_pct', 0)
            st.metric("Max Drawdown %", f"{abs(max_dd_pct):.1f}%")

        with col3:
            streaks = stats.get('streaks', {})
            st.metric("Max Win Streak", streaks.get('max_win_streak', 0))

        with col4:
            st.metric("Max Loss Streak", streaks.get('max_loss_streak', 0))

        st.markdown("---")

        # Drawdown chart
        st.subheader("Drawdown Analysis")
        fig_drawdown = create_drawdown_chart(df)
        st.plotly_chart(fig_drawdown, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Value at Risk")
            fig_var = create_var_chart(df)
            st.plotly_chart(fig_var, use_container_width=True)

        with col2:
            st.subheader("Streak Analysis")
            fig_streak = create_streak_chart(stats)
            st.plotly_chart(fig_streak, use_container_width=True)

        st.markdown("---")

        st.subheader("Betting Behavior After Results")
        fig_after = create_bet_after_result_chart(df, stats)
        st.plotly_chart(fig_after, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Multiplier Distribution")
            fig_mult = create_multiplier_distribution_chart(df)
            st.plotly_chart(fig_mult, use_container_width=True)

        with col2:
            st.subheader("Win Rate Trend")
            fig_wr = create_win_rate_trend_chart(df)
            st.plotly_chart(fig_wr, use_container_width=True)

        # Big wins timeline
        st.subheader("Big Wins Timeline")
        threshold = st.slider("Minimum multiplier", 2.0, 50.0, 5.0, 1.0)
        fig_bigwins = create_big_wins_timeline_chart(df, threshold)
        st.plotly_chart(fig_bigwins, use_container_width=True)

    # -------------------------------------------------------------------------
    # DATA TAB
    # -------------------------------------------------------------------------

    with tab5:
        st.subheader("Raw Data")

        # Display columns
        display_cols = [
            'timestamp', 'game', 'bet_amount', 'payout_amount',
            'profit', 'multiplier', 'is_win'
        ]
        available_cols = [c for c in display_cols if c in df.columns]

        # Search/filter
        search = st.text_input("🔍 Search", placeholder="Filter by game name...")

        display_df = df[available_cols].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        if search:
            display_df = display_df[
                display_df['game'].str.contains(search.lower(), case=False)
            ]

        # Show dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=500,
        )

        # Export options
        st.markdown("---")
        st.subheader("Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "betting_data.csv",
                "text/csv",
            )

        with col2:
            json_str = df.to_json(orient='records', date_format='iso')
            st.download_button(
                "📥 Download JSON",
                json_str,
                "betting_data.json",
                "application/json",
            )

        with col3:
            # Stats summary
            stats_df = pd.DataFrame([{
                'Total Bets': stats.get('total_bets', 0),
                'Total Wagered': stats.get('total_wagered', 0),
                'Total Payouts': stats.get('total_payouts', 0),
                'Net Profit': stats.get('net_profit', 0),
                'Win Rate': stats.get('win_rate', 0),
                'RTP': stats.get('rtp', 0),
                'Max Drawdown': stats.get('max_drawdown', 0),
            }])
            stats_csv = stats_df.to_csv(index=False)
            st.download_button(
                "📥 Download Stats",
                stats_csv,
                "betting_stats.csv",
                "text/csv",
            )

else:
    # =========================================================================
    # LANDING PAGE (NO DATA)
    # =========================================================================

    st.title("🎰 Betting Analytics Dashboard")
    st.markdown("---")

    st.markdown("""
    ### Welcome!

    Upload your bet archive JSON file(s) in the sidebar to get started.

    This dashboard provides comprehensive analysis of your crypto casino betting history including:

    - **📊 Financial Overview** - Cumulative P&L, game breakdown, daily heatmaps
    - **🎮 Game Analysis** - Deep dive into specific games (Plinko, Keno, etc.)
    - **⏰ Behavior Patterns** - Betting frequency, timing patterns, sessions
    - **⚠️ Risk Metrics** - Drawdown analysis, VaR, streak analysis
    - **💡 Auto Insights** - AI-generated observations about your betting patterns
    """)

    st.info("""
    **Supported format:** JSON export from crypto casino bet history

    The JSON should contain bet records with fields like:
    - `amount` - Bet amount
    - `payout` - Payout amount
    - `payoutMultiplier` - Win multiplier
    - `gameName` - Game played
    - `createdAt` - Timestamp
    """)

    # Show sample data option
    if st.button("📂 Load Sample Data"):
        try:
            with open('sample_data.json', 'r') as f:
                sample_content = f.read()

            raw_df = load_json_data(sample_content)
            st.session_state.raw_df = preprocess_data(raw_df)
            st.session_state.df = st.session_state.raw_df
            st.session_state.stats = calculate_statistics(st.session_state.df)
            st.session_state.file_info = {
                'files': 1,
                'total_bets': len(st.session_state.df),
                'date_range': (
                    st.session_state.df['timestamp'].min(),
                    st.session_state.df['timestamp'].max()
                ),
            }
            st.rerun()
        except FileNotFoundError:
            st.warning("Sample data file not found. Please upload your own data.")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("Betting Analytics Dashboard | Built with Streamlit")
