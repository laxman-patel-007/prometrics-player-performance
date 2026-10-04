"""
Cricket Data Pipeline: Real IPL Ball-by-Ball Processing (2008-2024)
Processes 260,000+ deliveries across 1,000+ matches to engineer 
professional player-level telemetry, composite indices, ratings, and auction valuations.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DELIV_PATH = os.path.join(BASE_DIR, "data/raw/deliveries_2008_2024.csv")
RAW_MATCH_PATH = os.path.join(BASE_DIR, "data/raw/matches_2008_2024.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/cricket_players_clean.csv")

def process_cricket_data():
    print(f"Loading deliveries dataset from {RAW_DELIV_PATH}...")
    df_deliv = pd.read_csv(RAW_DELIV_PATH)
    df_deliv.columns = [c.strip() for c in df_deliv.columns]
    
    print(f"Loading matches dataset from {RAW_MATCH_PATH}...")
    df_matches = pd.read_csv(RAW_MATCH_PATH)
    df_matches.columns = [c.strip() for c in df_matches.columns]
    
    # Strip string fields
    for col in ['batter', 'bowler', 'non_striker', 'batting_team', 'bowling_team', 'dismissal_kind', 'player_dismissed']:
        if col in df_deliv.columns:
            df_deliv[col] = df_deliv[col].astype(str).str.strip()
            
    # Clean matches metadata
    df_matches['player_of_match'] = df_matches['player_of_match'].astype(str).str.strip()
    mom_counts = df_matches['player_of_match'].value_counts().to_dict()

    print(f"Processing {len(df_deliv):,} delivery records across {len(df_matches)} matches...")

    # 1. Batting Statistics Aggregation
    df_deliv['is_dot_bat'] = (df_deliv['batsman_runs'] == 0) & (~df_deliv['extras_type'].isin(['wides']))
    df_deliv['is_four'] = df_deliv['batsman_runs'] == 4
    df_deliv['is_six'] = df_deliv['batsman_runs'] == 6
    df_deliv['is_boundary_run'] = df_deliv['is_four'] * 4 + df_deliv['is_six'] * 6

    # Phase categorization: Powerplay (overs 0-5), Middle (6-14), Death (15-19)
    df_deliv['phase'] = pd.cut(df_deliv['over'], bins=[-1, 5, 14, 20], labels=['Powerplay', 'Middle', 'Death'])

    # Aggregate by match and batter to compute innings, 30s, 50s, 100s
    batter_match = df_deliv.groupby(['batter', 'match_id']).agg(
        match_runs=('batsman_runs', 'sum'),
        match_balls=('batsman_runs', 'count')
    ).reset_index()

    batter_match_summary = batter_match.groupby('batter').agg(
        innings_batted=('match_id', 'count'),
        scores_30_plus=('match_runs', lambda x: (x >= 30).sum()),
        scores_50_plus=('match_runs', lambda x: (x >= 50).sum()),
        scores_100_plus=('match_runs', lambda x: (x >= 100).sum()),
        highest_score=('match_runs', 'max')
    ).reset_index()

    # Overall batting stats
    bat_stats = df_deliv[~df_deliv['extras_type'].isin(['wides'])].groupby('batter').agg(
        total_runs=('batsman_runs', 'sum'),
        balls_faced=('batsman_runs', 'count'),
        fours=('is_four', 'sum'),
        sixes=('is_six', 'sum'),
        boundary_runs=('is_boundary_run', 'sum'),
        dot_balls_faced=('is_dot_bat', 'sum')
    ).reset_index()

    # Dismissals count
    dismissals = df_deliv[df_deliv['player_dismissed'].notnull() & (df_deliv['player_dismissed'] != 'NA') & (df_deliv['player_dismissed'] != 'nan')]
    dismissal_counts = dismissals['player_dismissed'].value_counts().reset_index()
    dismissal_counts.columns = ['batter', 'dismissals']

    # Phase batting stats
    pp_bat = df_deliv[(df_deliv['phase'] == 'Powerplay') & (~df_deliv['extras_type'].isin(['wides']))].groupby('batter').agg(
        pp_runs=('batsman_runs', 'sum'),
        pp_balls=('batsman_runs', 'count')
    ).reset_index()
    pp_bat['powerplay_strike_rate'] = (pp_bat['pp_runs'] / np.maximum(pp_bat['pp_balls'], 1) * 100).round(2)

    death_bat = df_deliv[(df_deliv['phase'] == 'Death') & (~df_deliv['extras_type'].isin(['wides']))].groupby('batter').agg(
        death_runs=('batsman_runs', 'sum'),
        death_balls=('batsman_runs', 'count')
    ).reset_index()
    death_bat['death_overs_strike_rate'] = (death_bat['death_runs'] / np.maximum(death_bat['death_balls'], 1) * 100).round(2)

    # Merge batting components
    bat_df = bat_stats.merge(batter_match_summary, on='batter', how='left')
    bat_df = bat_df.merge(dismissal_counts, on='batter', how='left').fillna({'dismissals': 0})
    bat_df = bat_df.merge(pp_bat[['batter', 'powerplay_strike_rate']], on='batter', how='left').fillna({'powerplay_strike_rate': 100.0})
    bat_df = bat_df.merge(death_bat[['batter', 'death_overs_strike_rate']], on='batter', how='left').fillna({'death_overs_strike_rate': 120.0})

    # Batting metrics
    bat_df['batting_average'] = np.where(bat_df['dismissals'] > 0, 
                                         (bat_df['total_runs'] / bat_df['dismissals']).round(2), 
                                         bat_df['total_runs'].astype(float)).round(2)
    bat_df['batting_strike_rate'] = (bat_df['total_runs'] / np.maximum(bat_df['balls_faced'], 1) * 100).round(2)
    bat_df['boundary_run_pct'] = (bat_df['boundary_runs'] / np.maximum(bat_df['total_runs'], 1) * 100).round(2)
    bat_df['dot_ball_faced_pct'] = (bat_df['dot_balls_faced'] / np.maximum(bat_df['balls_faced'], 1) * 100).round(2)

    print(f"Aggregated batting statistics for {len(bat_df)} batsmen.")

    # 2. Bowling Statistics Aggregation
    bowler_valid_balls = df_deliv[~df_deliv['extras_type'].isin(['wides', 'noballs'])]
    df_deliv['bowler_runs_conceded'] = df_deliv['batsman_runs'] + df_deliv['extra_runs'].where(df_deliv['extras_type'].isin(['wides', 'noballs']), 0)
    
    # Wickets credited to bowler
    bowler_wickets = df_deliv[df_deliv['is_wicket'] == 1]
    bowler_wickets = bowler_wickets[~bowler_wickets['dismissal_kind'].isin(['run out', 'retired hurt', 'retired out', 'obstructing the field', 'NA', 'nan'])]
    
    wickets_agg = bowler_wickets.groupby('bowler')['is_wicket'].sum().reset_index()
    wickets_agg.columns = ['bowler', 'wickets_taken']

    # Bowler deliveries & runs
    bowl_balls = bowler_valid_balls.groupby('bowler')['ball'].count().reset_index()
    bowl_balls.columns = ['bowler', 'balls_bowled']
    
    bowl_runs = df_deliv.groupby('bowler')['bowler_runs_conceded'].sum().reset_index()
    bowl_runs.columns = ['bowler', 'runs_conceded']

    bowl_dots = df_deliv[(df_deliv['total_runs'] == 0) & (~df_deliv['extras_type'].isin(['wides', 'noballs']))].groupby('bowler')['ball'].count().reset_index()
    bowl_dots.columns = ['bowler', 'dot_balls_bowled']

    # Match-level bowling for innings and 3+ wicket hauls
    bowler_match = bowler_wickets.groupby(['bowler', 'match_id'])['is_wicket'].sum().reset_index()
    hauls_3w = bowler_match[bowler_match['is_wicket'] >= 3].groupby('bowler')['is_wicket'].count().reset_index()
    hauls_3w.columns = ['bowler', 'three_plus_wickets']

    bowl_innings = df_deliv.groupby('bowler')['match_id'].nunique().reset_index()
    bowl_innings.columns = ['bowler', 'innings_bowled']

    # Phase bowling (Death Overs: overs 16-20)
    death_bowl = df_deliv[df_deliv['over'] >= 15].groupby('bowler').agg(
        death_runs_conceded=('bowler_runs_conceded', 'sum'),
        death_balls_bowled=('ball', lambda x: (~df_deliv.loc[x.index, 'extras_type'].isin(['wides', 'noballs'])).sum()),
        death_wickets=('is_wicket', lambda x: (df_deliv.loc[x.index, 'is_wicket'] == 1).sum())
    ).reset_index()
    death_bowl['death_overs_economy'] = np.where(
        death_bowl['death_balls_bowled'] >= 12,
        (death_bowl['death_runs_conceded'] / (death_bowl['death_balls_bowled'] / 6.0)).round(2),
        10.5
    )

    # Merge bowling stats
    bowl_df = bowl_balls.merge(bowl_runs, on='bowler', how='left')
    bowl_df = bowl_df.merge(wickets_agg, on='bowler', how='left').fillna({'wickets_taken': 0})
    bowl_df = bowl_df.merge(bowl_dots, on='bowler', how='left').fillna({'dot_balls_bowled': 0})
    bowl_df = bowl_df.merge(bowl_innings, on='bowler', how='left').fillna({'innings_bowled': 0})
    bowl_df = bowl_df.merge(hauls_3w, on='bowler', how='left').fillna({'three_plus_wickets': 0})
    bowl_df = bowl_df.merge(death_bowl[['bowler', 'death_overs_economy']], on='bowler', how='left').fillna({'death_overs_economy': 9.8})

    # Bowling metrics
    bowl_df['overs_bowled'] = (bowl_df['balls_bowled'] / 6.0).round(1)
    bowl_df['economy_rate'] = np.where(bowl_df['balls_bowled'] >= 30, 
                                       (bowl_df['runs_conceded'] / (bowl_df['balls_bowled'] / 6.0)).round(2), 
                                       9.5)
    bowl_df['bowling_strike_rate'] = np.where(bowl_df['wickets_taken'] > 0, 
                                             (bowl_df['balls_bowled'] / bowl_df['wickets_taken']).round(2), 
                                             45.0)
    bowl_df['bowling_average'] = np.where(bowl_df['wickets_taken'] > 0, 
                                          (bowl_df['runs_conceded'] / bowl_df['wickets_taken']).round(2), 
                                          55.0)
    bowl_df['dot_ball_bowled_pct'] = (bowl_df['dot_balls_bowled'] / np.maximum(bowl_df['balls_bowled'], 1) * 100).round(2)

    print(f"Aggregated bowling statistics for {len(bowl_df)} bowlers.")

    # 3. Master Player Merge
    all_players = sorted(list(set(bat_df['batter']).union(set(bowl_df['bowler']))))
    
    # Calculate matches played per player
    batter_matches = df_deliv.groupby('batter')['match_id'].nunique().to_dict()
    bowler_matches = df_deliv.groupby('bowler')['match_id'].nunique().to_dict()

    players_list = []
    for player in all_players:
        b_m = batter_matches.get(player, 0)
        bw_m = bowler_matches.get(player, 0)
        total_m = max(b_m, bw_m)

        # Skip players with fewer than 3 matches to keep analysis robust
        if total_m < 3:
            continue

        b_row = bat_df[bat_df['batter'] == player]
        bw_row = bowl_df[bowl_df['bowler'] == player]

        # Batting inputs
        runs = int(b_row['total_runs'].values[0]) if len(b_row) > 0 else 0
        balls = int(b_row['balls_faced'].values[0]) if len(b_row) > 0 else 0
        bat_avg = float(b_row['batting_average'].values[0]) if len(b_row) > 0 else 5.0
        bat_sr = float(b_row['batting_strike_rate'].values[0]) if len(b_row) > 0 else 60.0
        b_fours = int(b_row['fours'].values[0]) if len(b_row) > 0 else 0
        b_sixes = int(b_row['sixes'].values[0]) if len(b_row) > 0 else 0
        bound_pct = float(b_row['boundary_run_pct'].values[0]) if len(b_row) > 0 else 0.0
        dot_bat_pct = float(b_row['dot_ball_faced_pct'].values[0]) if len(b_row) > 0 else 60.0
        high_score = int(b_row['highest_score'].values[0]) if len(b_row) > 0 else 0
        fifties = int(b_row['scores_50_plus'].values[0]) if len(b_row) > 0 else 0
        thirties = int(b_row['scores_30_plus'].values[0]) if len(b_row) > 0 else 0
        death_sr = float(b_row['death_overs_strike_rate'].values[0]) if len(b_row) > 0 else 100.0

        # Bowling inputs
        overs = float(bw_row['overs_bowled'].values[0]) if len(bw_row) > 0 else 0.0
        wickets = int(bw_row['wickets_taken'].values[0]) if len(bw_row) > 0 else 0
        econ = float(bw_row['economy_rate'].values[0]) if len(bw_row) > 0 else 10.5
        bowl_sr = float(bw_row['bowling_strike_rate'].values[0]) if len(bw_row) > 0 else 50.0
        bowl_avg = float(bw_row['bowling_average'].values[0]) if len(bw_row) > 0 else 60.0
        dot_bowl_pct = float(bw_row['dot_ball_bowled_pct'].values[0]) if len(bw_row) > 0 else 25.0
        hauls_3 = int(bw_row['three_plus_wickets'].values[0]) if len(bw_row) > 0 else 0
        death_econ = float(bw_row['death_overs_economy'].values[0]) if len(bw_row) > 0 else 11.0

        # Player of the Match awards
        mom = mom_counts.get(player, 0)

        # Primary Role Determination
        if wickets >= 20 and runs >= 600:
            role = "All-Rounder"
        elif overs >= 25 and wickets >= 15:
            role = "Specialist Bowler"
        elif runs >= 350 or balls >= 250:
            role = "Specialist Batter"
        elif overs >= 10:
            role = "Bowling Specialist"
        else:
            role = "Squad Batter"

        players_list.append({
            'player_name': player,
            'matches_played': total_m,
            'primary_role': role,
            'total_runs': runs,
            'balls_faced': balls,
            'batting_average': bat_avg,
            'batting_strike_rate': bat_sr,
            'fours': b_fours,
            'sixes': b_sixes,
            'boundary_run_pct': bound_pct,
            'dot_ball_faced_pct': dot_bat_pct,
            'highest_score': high_score,
            'thirties': thirties,
            'fifties': fifties,
            'death_overs_strike_rate': death_sr,
            'overs_bowled': overs,
            'wickets_taken': wickets,
            'economy_rate': econ,
            'bowling_strike_rate': bowl_sr,
            'bowling_average': bowl_avg,
            'dot_ball_bowled_pct': dot_bowl_pct,
            'three_plus_wickets': hauls_3,
            'death_overs_economy': death_econ,
            'player_of_match_awards': mom
        })

    df_players = pd.DataFrame(players_list)
    print(f"Total qualified players identified: {len(df_players)}")

    # 4. Composite Factor Indices (Sports Science & Franchise Analytics Formulae)
    # Normalized Batting Factor Score (0 to 100)
    bat_score = (
        np.clip(df_players['batting_average'] / 45.0, 0, 1.5) * 35.0 +
        np.clip(df_players['batting_strike_rate'] / 160.0, 0, 1.5) * 35.0 +
        np.clip(df_players['boundary_run_pct'] / 75.0, 0, 1.5) * 15.0 +
        np.clip(df_players['death_overs_strike_rate'] / 200.0, 0, 1.5) * 15.0
    ).round(2)

    # Normalized Bowling Factor Score (0 to 100)
    # Lower economy and lower strike rate are superior
    econ_factor = np.clip((11.0 - df_players['economy_rate']) / 4.0, 0, 1.5) * 40.0
    bowl_sr_factor = np.clip((35.0 - df_players['bowling_strike_rate']) / 18.0, 0, 1.5) * 35.0
    dots_factor = np.clip(df_players['dot_ball_bowled_pct'] / 50.0, 0, 1.5) * 25.0
    bowl_score = np.where(
        df_players['overs_bowled'] >= 10,
        (econ_factor + bowl_sr_factor + dots_factor).round(2),
        15.0
    )

    df_players['batting_impact_index'] = bat_score
    df_players['bowling_impact_index'] = bowl_score
    df_players['clutch_match_winner_index'] = (
        np.clip(df_players['player_of_match_awards'] * 4.0, 0, 40) + 
        np.clip(df_players['fifties'] * 2.5, 0, 30) + 
        np.clip(df_players['three_plus_wickets'] * 3.0, 0, 30)
    ).round(2)

    # Overall Continuous Performance Rating (Scale 50.0 to 95.0)
    # Role-adjusted weighting
    ratings = []
    for _, row in df_players.iterrows():
        r = row['primary_role']
        exp_boost = min(row['matches_played'] / 150.0, 1.0) * 8.0 # Experience factor
        mom_boost = min(row['player_of_match_awards'] * 0.45, 6.0)

        if "All-Rounder" in r:
            base_score = 0.50 * row['batting_impact_index'] + 0.50 * row['bowling_impact_index']
        elif "Bowler" in r:
            base_score = 0.85 * row['bowling_impact_index'] + 0.15 * row['batting_impact_index']
        else: # Batter
            base_score = 0.85 * row['batting_impact_index'] + 0.15 * row['bowling_impact_index']

        overall = 52.0 + (base_score * 0.35) + exp_boost + mom_boost
        ratings.append(round(float(np.clip(overall, 50.0, 94.8)), 1))

    df_players['overall_performance_rating'] = ratings

    # Talent Tier Classification (0: Developing/Squad, 1: Core/Star, 2: Elite/Marquee)
    tiers = []
    tier_codes = []
    for rating in df_players['overall_performance_rating']:
        if rating >= 80.0:
            tiers.append("Elite / Marquee")
            tier_codes.append(2)
        elif rating >= 68.0:
            tiers.append("Core / Star")
            tier_codes.append(1)
        else:
            tiers.append("Developing / Squad")
            tier_codes.append(0)

    df_players['performance_tier'] = tiers
    df_players['performance_tier_code'] = tier_codes

    # Estimated IPL Auction Valuation in ₹ Crores (based on rating, role, and tier)
    auction_vals = []
    for _, row in df_players.iterrows():
        val = np.exp((row['overall_performance_rating'] - 58.0) * 0.09) * 1.5
        if row['primary_role'] == 'All-Rounder':
            val *= 1.25 # All-rounders command a premium at IPL auctions
        if row['player_of_match_awards'] >= 10:
            val *= 1.15
        auction_vals.append(round(float(np.clip(val, 0.5, 24.5)), 1))

    df_players['estimated_auction_val_cr'] = auction_vals

    # Ensure processed directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_players.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed Cricket Dataset saved successfully to {PROCESSED_DATA_PATH}")
    print(f"Dataset Shape: {df_players.shape}")
    print(df_players[['player_name', 'primary_role', 'matches_played', 'total_runs', 'wickets_taken', 'overall_performance_rating', 'performance_tier', 'estimated_auction_val_cr']].head(10))

if __name__ == "__main__":
    process_cricket_data()
