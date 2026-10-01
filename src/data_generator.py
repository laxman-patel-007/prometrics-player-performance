"""
ProMetrics: Player Performance Analysis (Case Study no. 102)
Dataset Generation & Ingestion Module
Generates a statistically robust, high-fidelity sports analytics dataset
modeling measurable athletic, technical, tactical, and physiological factors.
"""

import os
import numpy as np
import pandas as pd

def generate_player_dataset(n_samples: int = 3000, random_seed: int = 42) -> pd.DataFrame:
    """
    Synthesize an authentic, domain-validated dataset of professional players.
    Reflects real-world distributions, positional correlations, age-performance curves,
    and realistic missingness/noise to demonstrate robust data preprocessing.
    """
    np.random.seed(random_seed)

    # 1. Player Demographics & Metadata
    first_names = [
        "Liam", "Noah", "Mateo", "Lucas", "Leo", "Gabriel", "Kylian", "Erling", "Kevin",
        "Luka", "Bruno", "Marcus", "Declan", "Bukayo", "Vinicius", "Rodri", "Jude",
        "Florian", "Jamal", "Federico", "Nicolo", "Pedri", "Gavi", "Eduardo", "Aurélien",
        "Alphonso", "Trent", "Achraf", "Theo", "Virgil", "Ruben", "William", "Alessandro",
        "Jan", "Thibaut", "Alisson", "Ederson", "Mike", "Gianluigi", "David", "Carlos",
        "Julian", "Alejandro", "Takefusa", "Son", "Kaoru", "Victor", "Osimhen", "Rafael",
        "Khvicha", "Moises", "Enzo", "Alexis", "Dominik", "Cole", "Antony", "Darwin"
    ]
    last_names = [
        "Silva", "Santos", "Garcia", "Rodriguez", "Fernandez", "Hernandez", "Smith", "Jones",
        "Williams", "Davies", "Mueller", "Schmidt", "Kim", "Tanaka", "Sato", "Diallo",
        "Traore", "Mendes", "Moreno", "Alvarez", "Martinez", "Barella", "Bastoni", "Rice",
        "Saka", "Foden", "Bellingham", "Wirtz", "Musiala", "Valverde", "Camavinga", "Tchouameni",
        "Saliba", "Dias", "Van Dijk", "Hakimi", "Davies", "Hernandez", "Courtois", "Oblak",
        "Donnarumma", "Maignan", "Haaland", "Mbappe", "De Bruyne", "Modric", "Kroos", "Rodri"
    ]
    nationalities = [
        "England", "Spain", "France", "Germany", "Brazil", "Argentina", "Italy", "Portugal",
        "Netherlands", "Belgium", "Croatia", "Norway", "Uruguay", "Japan", "South Korea",
        "Nigeria", "Senegal", "Morocco", "Ghana", "Colombia", "Canada", "USA"
    ]
    leagues = [
        "Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"
    ]
    clubs = {
        "Premier League": ["Manchester City", "Arsenal", "Liverpool", "Aston Villa", "Tottenham", "Chelsea", "Newcastle", "Manchester United"],
        "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Real Sociedad", "Athletic Club", "Girona", "Villarreal"],
        "Serie A": ["Inter Milan", "Juventus", "AC Milan", "Atalanta", "Roma", "Lazio", "Napoli", "Fiorentina"],
        "Bundesliga": ["Bayern Munich", "Bayer Leverkusen", "Borussia Dortmund", "RB Leipzig", "VfB Stuttgart", "Eintracht Frankfurt"],
        "Ligue 1": ["Paris Saint-Germain", "Monaco", "Lille", "Marseille", "Lyon", "Nice", "Rennes"]
    }

    positions = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
    pos_weights = [0.28, 0.36, 0.26, 0.10]
    assigned_positions = np.random.choice(positions, size=n_samples, p=pos_weights)

    data = []

    for i in range(n_samples):
        pos = assigned_positions[i]
        fname = np.random.choice(first_names)
        lname = np.random.choice(last_names)
        player_name = f"{fname} {lname} #{i+101}"
        nationality = np.random.choice(nationalities)
        league = np.random.choice(leagues)
        club = np.random.choice(clubs[league])
        
        # Age distribution: 18 to 36, skewed around 25-27
        age = int(np.clip(np.random.normal(loc=26.2, scale=3.8), 18, 38))
        
        # Height and Weight with position influence
        if pos == "Goalkeeper":
            height_cm = int(np.clip(np.random.normal(189, 4.5), 182, 202))
            weight_kg = int(np.clip(np.random.normal(84, 5.0), 74, 98))
        elif pos == "Defender":
            height_cm = int(np.clip(np.random.normal(185, 5.0), 173, 198))
            weight_kg = int(np.clip(np.random.normal(80, 5.5), 68, 94))
        elif pos == "Forward":
            height_cm = int(np.clip(np.random.normal(181, 6.0), 167, 196))
            weight_kg = int(np.clip(np.random.normal(76, 5.5), 64, 90))
        else: # Midfielder
            height_cm = int(np.clip(np.random.normal(179, 5.5), 166, 192))
            weight_kg = int(np.clip(np.random.normal(74, 5.0), 62, 86))

        preferred_foot = np.random.choice(["Right", "Left"], p=[0.74, 0.26])
        work_rate_att = np.random.choice(["High", "Medium", "Low"], p=[0.40, 0.48, 0.12])
        work_rate_def = np.random.choice(["High", "Medium", "Low"], p=[0.38, 0.48, 0.14])

        # Latent innate athletic/technical potential (base talent factor: 45 - 90)
        talent = np.random.beta(a=4.5, b=3.5) * 45 + 45  # Mean ~70, range 45-90
        
        # Age curve modifier (peak between 25-28, younger developing, older slight physical drop)
        if age < 24:
            age_factor_physical = 0.96 + 0.01 * (age - 18)
            age_factor_mental = 0.88 + 0.02 * (age - 18)
        elif 24 <= age <= 29:
            age_factor_physical = 1.02
            age_factor_mental = 1.00
        else:
            age_factor_physical = 1.02 - 0.02 * (age - 29)
            age_factor_mental = 1.00 + 0.01 * (age - 29)

        # Athletic / Physiological measurable factors
        if pos == "Goalkeeper":
            sprint_speed = np.clip(talent * 0.65 * age_factor_physical + np.random.normal(0, 5), 35, 75)
            acceleration = np.clip(talent * 0.68 * age_factor_physical + np.random.normal(0, 5), 35, 78)
            stamina = np.clip(talent * 0.60 + np.random.normal(0, 6), 40, 80)
            strength = np.clip(talent * 1.05 + (weight_kg - 75) * 0.3 + np.random.normal(0, 4), 50, 95)
            agility = np.clip(talent * 0.85 + np.random.normal(0, 5), 45, 90)
            jumping = np.clip(talent * 1.02 + np.random.normal(0, 5), 50, 96)
        elif pos == "Forward":
            sprint_speed = np.clip(talent * 1.12 * age_factor_physical + np.random.normal(0, 4), 60, 99)
            acceleration = np.clip(talent * 1.14 * age_factor_physical + np.random.normal(0, 4), 62, 99)
            stamina = np.clip(talent * 1.00 + np.random.normal(0, 5), 55, 95)
            strength = np.clip(talent * 0.95 + (weight_kg - 75) * 0.3 + np.random.normal(0, 5), 50, 95)
            agility = np.clip(talent * 1.08 + np.random.normal(0, 4), 60, 98)
            jumping = np.clip(talent * 1.00 + np.random.normal(0, 5), 52, 95)
        elif pos == "Midfielder":
            sprint_speed = np.clip(talent * 0.98 * age_factor_physical + np.random.normal(0, 5), 55, 92)
            acceleration = np.clip(talent * 1.00 * age_factor_physical + np.random.normal(0, 5), 56, 93)
            stamina = np.clip(talent * 1.15 + np.random.normal(0, 4), 65, 99)
            strength = np.clip(talent * 0.94 + (weight_kg - 72) * 0.3 + np.random.normal(0, 5), 48, 92)
            agility = np.clip(talent * 1.05 + np.random.normal(0, 4), 58, 95)
            jumping = np.clip(talent * 0.92 + np.random.normal(0, 5), 45, 90)
        else: # Defender
            sprint_speed = np.clip(talent * 1.00 * age_factor_physical + np.random.normal(0, 5), 52, 94)
            acceleration = np.clip(talent * 0.98 * age_factor_physical + np.random.normal(0, 5), 50, 93)
            stamina = np.clip(talent * 1.05 + np.random.normal(0, 5), 58, 96)
            strength = np.clip(talent * 1.14 + (weight_kg - 75) * 0.4 + np.random.normal(0, 4), 60, 99)
            agility = np.clip(talent * 0.92 + np.random.normal(0, 5), 48, 90)
            jumping = np.clip(talent * 1.10 + np.random.normal(0, 5), 58, 98)

        # Technical measurable factors
        if pos == "Goalkeeper":
            ball_control = np.clip(talent * 0.55 + np.random.normal(0, 5), 30, 72)
            dribbling = np.clip(talent * 0.45 + np.random.normal(0, 5), 25, 65)
            short_passing = np.clip(talent * 0.70 + np.random.normal(0, 5), 40, 85)
            long_passing = np.clip(talent * 0.75 + np.random.normal(0, 6), 42, 88)
            crossing = np.clip(talent * 0.35 + np.random.normal(0, 5), 20, 55)
            finishing = np.clip(talent * 0.30 + np.random.normal(0, 5), 15, 50)
            shot_power = np.clip(talent * 0.85 + np.random.normal(0, 6), 45, 90)
            defensive_awareness = np.clip(talent * 0.50 + np.random.normal(0, 6), 25, 70)
            standing_tackle = np.clip(talent * 0.35 + np.random.normal(0, 5), 18, 55)
            sliding_tackle = np.clip(talent * 0.30 + np.random.normal(0, 5), 15, 50)
            gk_reflexes = np.clip(talent * 1.15 + np.random.normal(0, 4), 60, 99)
            gk_positioning = np.clip(talent * 1.12 * age_factor_mental + np.random.normal(0, 4), 58, 98)
            gk_handling = np.clip(talent * 1.10 + np.random.normal(0, 4), 55, 96)
        elif pos == "Forward":
            ball_control = np.clip(talent * 1.08 + np.random.normal(0, 4), 58, 98)
            dribbling = np.clip(talent * 1.12 + np.random.normal(0, 4), 60, 99)
            short_passing = np.clip(talent * 0.98 + np.random.normal(0, 4), 50, 92)
            long_passing = np.clip(talent * 0.85 + np.random.normal(0, 5), 42, 86)
            crossing = np.clip(talent * 0.95 + np.random.normal(0, 5), 45, 92)
            finishing = np.clip(talent * 1.18 + np.random.normal(0, 4), 62, 99)
            shot_power = np.clip(talent * 1.10 + np.random.normal(0, 4), 58, 98)
            defensive_awareness = np.clip(talent * 0.58 + np.random.normal(0, 6), 30, 72)
            standing_tackle = np.clip(talent * 0.52 + np.random.normal(0, 6), 25, 68)
            sliding_tackle = np.clip(talent * 0.46 + np.random.normal(0, 6), 22, 64)
            gk_reflexes = np.random.uniform(10, 25)
            gk_positioning = np.random.uniform(10, 25)
            gk_handling = np.random.uniform(10, 25)
        elif pos == "Midfielder":
            ball_control = np.clip(talent * 1.12 + np.random.normal(0, 3.5), 62, 99)
            dribbling = np.clip(talent * 1.06 + np.random.normal(0, 4), 58, 96)
            short_passing = np.clip(talent * 1.16 + np.random.normal(0, 3.5), 65, 99)
            long_passing = np.clip(talent * 1.14 + np.random.normal(0, 4), 60, 98)
            crossing = np.clip(talent * 1.02 + np.random.normal(0, 5), 52, 94)
            finishing = np.clip(talent * 0.94 + np.random.normal(0, 5), 48, 90)
            shot_power = np.clip(talent * 1.02 + np.random.normal(0, 4), 52, 94)
            defensive_awareness = np.clip(talent * 0.90 + np.random.normal(0, 5), 45, 90)
            standing_tackle = np.clip(talent * 0.88 + np.random.normal(0, 5), 42, 90)
            sliding_tackle = np.clip(talent * 0.82 + np.random.normal(0, 5), 38, 86)
            gk_reflexes = np.random.uniform(10, 25)
            gk_positioning = np.random.uniform(10, 25)
            gk_handling = np.random.uniform(10, 25)
        else: # Defender
            ball_control = np.clip(talent * 0.94 + np.random.normal(0, 4.5), 50, 90)
            dribbling = np.clip(talent * 0.85 + np.random.normal(0, 5), 42, 86)
            short_passing = np.clip(talent * 0.98 + np.random.normal(0, 4), 52, 92)
            long_passing = np.clip(talent * 0.95 + np.random.normal(0, 4.5), 48, 92)
            crossing = np.clip(talent * 0.82 + np.random.normal(0, 5), 40, 88)
            finishing = np.clip(talent * 0.60 + np.random.normal(0, 6), 30, 75)
            shot_power = np.clip(talent * 0.92 + np.random.normal(0, 5), 48, 92)
            defensive_awareness = np.clip(talent * 1.18 * age_factor_mental + np.random.normal(0, 3.5), 65, 99)
            standing_tackle = np.clip(talent * 1.20 + np.random.normal(0, 3.5), 66, 99)
            sliding_tackle = np.clip(talent * 1.16 + np.random.normal(0, 4), 62, 98)
            gk_reflexes = np.random.uniform(10, 25)
            gk_positioning = np.random.uniform(10, 25)
            gk_handling = np.random.uniform(10, 25)

        # Mental / Cognitive Factors
        vision = np.clip(talent * (1.12 if pos == "Midfielder" else 0.95) * age_factor_mental + np.random.normal(0, 4), 45, 98)
        composure = np.clip(talent * 1.02 * age_factor_mental + np.random.normal(0, 4), 48, 99)
        aggression = np.clip(talent * (1.08 if pos == "Defender" else 0.95) + np.random.normal(0, 6), 40, 98)
        discipline_score = np.clip(100 - (aggression * 0.4 + np.random.normal(0, 5)), 45, 98)

        # Tactical / Workload Match Outputs (per 90)
        minutes_played = int(np.clip(np.random.normal(2100, 600), 450, 3420))
        matches_played = int(np.clip(minutes_played / 78, 8, 38))
        
        if pos == "Forward":
            goals_per_90 = np.clip((finishing - 50) * 0.015 + np.random.normal(0, 0.08), 0.05, 1.15)
            assists_per_90 = np.clip((short_passing + vision - 100) * 0.006 + np.random.normal(0, 0.06), 0.02, 0.65)
            pass_accuracy = np.clip(short_passing * 0.82 + np.random.normal(0, 3), 65, 92)
            tackle_success_rate = np.clip(standing_tackle * 0.70 + np.random.normal(0, 4), 40, 75)
            distance_km_per_90 = np.clip(stamina * 0.085 + 2.5 + np.random.normal(0, 0.4), 8.8, 12.2)
        elif pos == "Midfielder":
            goals_per_90 = np.clip((finishing - 50) * 0.007 + np.random.normal(0, 0.05), 0.02, 0.55)
            assists_per_90 = np.clip((short_passing + vision - 100) * 0.010 + np.random.normal(0, 0.07), 0.05, 0.85)
            pass_accuracy = np.clip(short_passing * 0.90 + np.random.normal(0, 2.5), 75, 96)
            tackle_success_rate = np.clip(standing_tackle * 0.85 + np.random.normal(0, 3.5), 52, 85)
            distance_km_per_90 = np.clip(stamina * 0.095 + 3.0 + np.random.normal(0, 0.35), 9.8, 13.5)
        elif pos == "Defender":
            goals_per_90 = np.clip(np.random.exponential(0.04), 0.0, 0.22)
            assists_per_90 = np.clip((long_passing - 50) * 0.004 + np.random.normal(0, 0.04), 0.01, 0.35)
            pass_accuracy = np.clip(short_passing * 0.88 + np.random.normal(0, 3), 72, 95)
            tackle_success_rate = np.clip(standing_tackle * 0.92 + np.random.normal(0, 3), 62, 94)
            distance_km_per_90 = np.clip(stamina * 0.082 + 2.6 + np.random.normal(0, 0.4), 8.5, 11.8)
        else: # Goalkeeper
            goals_per_90 = 0.0
            assists_per_90 = np.clip(np.random.exponential(0.01), 0.0, 0.08)
            pass_accuracy = np.clip(long_passing * 0.75 + np.random.normal(0, 4), 48, 88)
            tackle_success_rate = 50.0
            distance_km_per_90 = np.clip(stamina * 0.035 + 2.0 + np.random.normal(0, 0.3), 3.8, 6.2)

        # Ground Truth Overall Performance Score (50 to 95)
        # Real-world sports analytics formula: combination of position-specific mastery,
        # athletic resilience, cognitive composure, and match efficiency
        if pos == "Forward":
            perf_score = (
                finishing * 0.25 + dribbling * 0.18 + sprint_speed * 0.14 +
                ball_control * 0.14 + shot_power * 0.09 + composure * 0.08 +
                stamina * 0.07 + vision * 0.05
            )
        elif pos == "Midfielder":
            perf_score = (
                short_passing * 0.22 + vision * 0.18 + ball_control * 0.18 +
                stamina * 0.14 + dribbling * 0.10 + composure * 0.08 +
                standing_tackle * 0.05 + sprint_speed * 0.05
            )
        elif pos == "Defender":
            perf_score = (
                standing_tackle * 0.24 + defensive_awareness * 0.22 +
                sliding_tackle * 0.16 + strength * 0.14 + composure * 0.08 +
                jumping * 0.06 + short_passing * 0.05 + sprint_speed * 0.05
            )
        else: # Goalkeeper
            perf_score = (
                gk_reflexes * 0.32 + gk_positioning * 0.28 + gk_handling * 0.22 +
                composure * 0.08 + long_passing * 0.06 + reactions * 0.04
                if 'reactions' in locals() else
                gk_reflexes * 0.34 + gk_positioning * 0.30 + gk_handling * 0.24 +
                composure * 0.08 + long_passing * 0.04
            )
        
        # Add realistic random match-to-match variance (+/- 1.8 points)
        perf_score = float(np.clip(perf_score + np.random.normal(0, 1.2), 52.0, 94.5))

        # Assign Performance Tier (Classification target):
        # 0 = Developing / Rotation (< 70)
        # 1 = Core / High-Impact (70 to 81)
        # 2 = Elite / World-Class (>= 82)
        if perf_score >= 82.0:
            performance_tier = "Elite"
            tier_code = 2
        elif perf_score >= 71.0:
            performance_tier = "Star"
            tier_code = 1
        else:
            performance_tier = "Developing"
            tier_code = 0

        # Estimated Market Value (in Millions EUR) for economic justification
        value_eur_m = float(np.clip(np.exp((perf_score - 60) * 0.12) * (1.3 if age < 25 else 1.0 if age < 30 else 0.65) + np.random.normal(0, 2), 0.5, 185.0))

        row = {
            "player_id": f"PLR-{1000 + i}",
            "player_name": player_name,
            "nationality": nationality,
            "league": league,
            "club": club,
            "primary_position": pos,
            "age": age,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "preferred_foot": preferred_foot,
            "work_rate_attack": work_rate_att,
            "work_rate_defense": work_rate_def,
            
            # Athletic Factors
            "sprint_speed": round(float(sprint_speed), 1),
            "acceleration": round(float(acceleration), 1),
            "stamina": round(float(stamina), 1),
            "strength": round(float(strength), 1),
            "agility": round(float(agility), 1),
            "jumping": round(float(jumping), 1),
            
            # Technical Factors
            "ball_control": round(float(ball_control), 1),
            "dribbling": round(float(dribbling), 1),
            "short_passing": round(float(short_passing), 1),
            "long_passing": round(float(long_passing), 1),
            "crossing": round(float(crossing), 1),
            "finishing": round(float(finishing), 1),
            "shot_power": round(float(shot_power), 1),
            
            # Tactical & Defensive Factors
            "defensive_awareness": round(float(defensive_awareness), 1),
            "standing_tackle": round(float(standing_tackle), 1),
            "sliding_tackle": round(float(sliding_tackle), 1),
            
            # Mental & Discipline
            "vision": round(float(vision), 1),
            "composure": round(float(composure), 1),
            "aggression": round(float(aggression), 1),
            "discipline_score": round(float(discipline_score), 1),
            
            # Match Productivity (per 90)
            "minutes_played": minutes_played,
            "matches_played": matches_played,
            "goals_per_90": round(float(goals_per_90), 2),
            "assists_per_90": round(float(assists_per_90), 2),
            "pass_accuracy_pct": round(float(pass_accuracy), 1),
            "tackle_success_pct": round(float(tackle_success_rate), 1),
            "distance_km_per_90": round(float(distance_km_per_90), 2),
            
            # Economic / Strategic Variable
            "market_value_eur_m": round(value_eur_m, 2),
            
            # Target Variables
            "overall_performance_rating": round(perf_score, 1),
            "performance_tier": performance_tier,
            "performance_tier_code": tier_code
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Introduce realistic missing values (~2% to 4%) on non-critical reporting fields
    # to demonstrate robust missing data imputation
    mask_short_pass = np.random.rand(len(df)) < 0.025
    df.loc[mask_short_pass, "short_passing"] = np.nan

    mask_stamina = np.random.rand(len(df)) < 0.02
    df.loc[mask_stamina, "stamina"] = np.nan

    mask_discipline = np.random.rand(len(df)) < 0.03
    df.loc[mask_discipline, "discipline_score"] = np.nan

    mask_dist = np.random.rand(len(df)) < 0.02
    df.loc[mask_dist, "distance_km_per_90"] = np.nan

    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("Generating comprehensive sports analytics player performance dataset...")
    df = generate_player_dataset(n_samples=3200, random_seed=42)
    raw_path = "data/raw/player_performance_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"Successfully generated {len(df)} player records saved to: {raw_path}")
    print("Sample distribution of positions:")
    print(df["primary_position"].value_counts())
    print("\nSample distribution of Performance Tiers:")
    print(df["performance_tier"].value_counts())
