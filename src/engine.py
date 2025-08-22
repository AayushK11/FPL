import pandas as pd
import numpy as np
import pulp
import constants
import os
from rapidfuzz import process, fuzz

class FPLDataManager:
    def __init__(self, players_df, teams_df=None, fixtures_df=None):
        self.players_df = players_df.copy()
        self.teams_df = teams_df.copy() if teams_df is not None else None
        self.fixtures_df = fixtures_df.copy() if fixtures_df is not None else None

    def merge_xgxa(self, xgxa_df):
        if "id" in xgxa_df.columns and "id" in self.players_df.columns:
            self.players_df["id"] = self.players_df["id"].astype(int)
            xgxa_df["id"] = xgxa_df["id"].astype(int)
            self.players_df = self.players_df.merge(xgxa_df[["id", "xG", "xA"]], on="id", how="left")
        else:
            self.players_df = self.players_df.merge(xgxa_df[["web_name", "xG", "xA"]], on="web_name", how="left")
        self.players_df["xG"] = self.players_df["xG"].fillna(0.0)
        self.players_df["xA"] = self.players_df["xA"].fillna(0.0)
        return self.players_df

    def clean(self):
        # Add more cleaning steps as needed
        self.players_df = self.players_df.copy()
        return self.players_df

class FPLOddsEngine:
    def __init__(self, players_df, fixtures_df=None, teams_df=None):
        self.players_df = players_df.copy()
        self.fixtures_df = fixtures_df.copy() if fixtures_df is not None else None
        self.teams_df = teams_df.copy() if teams_df is not None else None

    def generate_mock_xg_xa(self, seed=42):
        rng = np.random.default_rng(seed)
        rows = []
        for _, p in self.players_df.iterrows():
            minutes = p.get("minutes", 0)
            goals = p.get("goals_scored", 0)
            assists = p.get("assists", 0)
            if minutes >= 90 and (goals + assists) > 0:
                xG = max(0.01, (goals + rng.normal(0, 0.2)) / (minutes / 90))
                xA = max(0.0, (assists + rng.normal(0, 0.15)) / (minutes / 90))
            else:
                xG = float(rng.uniform(0, 0.2))
                xA = float(rng.uniform(0, 0.15))
            rows.append({"id": p["id"], "web_name": p["web_name"], "xG": xG, "xA": xA})
        return pd.DataFrame(rows)

    def compute_expected_points(self, xgxa_df=None, odds_df=None, params=None):
        players = self.players_df.copy()

        if xgxa_df is not None:
            xgxa_df["norm_name"] = xgxa_df["web_name"].str.replace(" ", "", regex=False).str.lower()
            xgxa_names = xgxa_df["norm_name"].tolist()
            def fetch_xgxa(row):
                norm_name = (str(row.get("first_name", "")).strip() + str(row.get("second_name", "")).strip()).replace(" ", "").lower()
                match, score, idx = process.extractOne(norm_name, xgxa_names, scorer=fuzz.ratio)
                if score >= 80:
                    stats = xgxa_df.iloc[idx]
                    return pd.Series([float(stats["xG"]), float(stats["xA"])] )
                else:
                    return pd.Series([0.0, 0.0])
            players[["xG", "xA"]] = players.apply(fetch_xgxa, axis=1)
            players["xG"] = pd.to_numeric(players["xG"], errors="coerce").fillna(0.0)
            players["xA"] = pd.to_numeric(players["xA"], errors="coerce").fillna(0.0)

        # Merge odds if available (optional)
        if odds_df is not None:
            if "id" in odds_df.columns and "id" in players.columns:
                players = players.merge(odds_df, on="id", how="left")
            else:
                players = players.merge(odds_df, on="web_name", how="left")
        # Minutes factor
        p = constants.POINTS_PARAM.copy()
        players["minutes_factor"] = (
            np.log1p(players.get("minutes", 0)) * p["minutes_scaler"]
        )
        # Offensive EP per 90
        # Use fillna(0) to avoid KeyError if any xG/xA values are missing
        if "xG" not in players.columns:
            players["xG"] = 0.0
        if "xA" not in players.columns:
            players["xA"] = 0.0
        players["ep_offense_per90"] = (
            players["xG"].fillna(0) * p["goal_points"] + players["xA"].fillna(0) * p["assist_points"]
        )
        # Bonus probability
        if "ict_index" in players.columns:
            players["ict_index"] = pd.to_numeric(players["ict_index"], errors="coerce")
            players["bonus_prob"] = (
                players["ict_index"].fillna(10) / (players["ict_index"].max() + 1)
            ) * 0.8 + p["bonus_base"] * 0.2
        else:
            players["bonus_prob"] = np.clip(
                players["xG"] / (players["xG"].max() + 1) * 0.6 + p["bonus_base"], 0, 1
            )
        players["ep_bonus"] = players["bonus_prob"] * 3

        # --- Aggregate fixture difficulty and EP for next 3 GWs ---
        if self.fixtures_df is not None and "kickoff_time" in self.fixtures_df.columns:
            # Ensure kickoff_time is datetime for comparison
            self.fixtures_df["kickoff_time"] = pd.to_datetime(self.fixtures_df["kickoff_time"], errors="coerce")
            now = pd.Timestamp.now(tz='UTC')
            future = self.fixtures_df[self.fixtures_df["kickoff_time"] >= now]
            future_sorted = future.sort_values("kickoff_time")
            # Only consider next 3 fixtures for each team
            home_diffs = (
                future_sorted.groupby("team_h")
                .head(3)[["team_h", "team_h_difficulty"]]
                .groupby("team_h")["team_h_difficulty"]
                .mean()
                .reset_index()
                .rename(columns={"team_h": "team", "team_h_difficulty": "avg_difficulty"})
            )
            away_diffs = (
                future_sorted.groupby("team_a")
                .head(3)[["team_a", "team_a_difficulty"]]
                .groupby("team_a")["team_a_difficulty"]
                .mean()
                .reset_index()
                .rename(columns={"team_a": "team", "team_a_difficulty": "avg_difficulty"})
            )
            avg_diffs = pd.concat([home_diffs, away_diffs], ignore_index=True)
            avg_diffs = avg_diffs.groupby("team")["avg_difficulty"].mean().reset_index()
            players = players.merge(avg_diffs, on="team", how="left")
            players["difficulty"] = players["avg_difficulty"]
            fixture_multiplier_adj = (6 - players["difficulty"].fillna(3)) / 5
            fixture_multiplier_adj = fixture_multiplier_adj.clip(
                lower=p.get("fixture_hard_penalty", 0.85),
                upper=p.get("fixture_easy_boost", 1.1),
            )
        else:
            fixture_multiplier_adj = 1.0

        # --- Team strength multiplier ---
        if self.teams_df is not None and "strength_overall_home" in self.teams_df.columns and "strength_overall_away" in self.teams_df.columns:
            self.teams_df["strength_avg"] = (
                self.teams_df["strength_overall_home"] + self.teams_df["strength_overall_away"]
            ) / 2
            strength_map = self.teams_df.set_index("id")["strength_avg"].to_dict()
        else:
            strength_map = {t["id"]: 1.0 for _, t in self.teams_df.iterrows()} if self.teams_df is not None else {}
        players["team_strength"] = players["team"].map(strength_map).fillna(1.0)

        # --- Odds multiplier ---
        if "goal_odds" in players.columns:
            players["goal_prob_from_odds"] = 1 / players["goal_odds"].replace(0, np.nan)
            players["odds_multiplier"] = (
                1
                + (
                    players["goal_prob_from_odds"].fillna(0)
                    - players["goal_prob_from_odds"].mean()
                ).fillna(0)
                * p["bookmaker_weight"]
            )
        else:
            players["odds_multiplier"] = 1.0

        # --- EP per 90 ---
        players["ep_per90"] = (
            (
                players["ep_offense_per90"]
                + players["ep_bonus"]
                + players["minutes_factor"]
            )
            * fixture_multiplier_adj
            * players["odds_multiplier"]
        )

        # --- Expected minutes next ---
        if "appearances" not in players:
            players["appearances"] = 1
        players["expected_minutes_next"] = np.where(
            players["minutes"] > 0,
            np.minimum(90, players["minutes"] / np.maximum(players["appearances"], 1)),
            60,
        )

        # --- Minutes reliability multiplier ---
        minutes_ratio = (players["minutes"] / players["appearances"].replace(0, 1)) / 90
        minutes_multiplier = minutes_ratio.clip(
            lower=p.get("minutes_floor", 0.5), upper=1.0
        )

        # --- Form multiplier ---
        if "form" in players.columns:
            players["form"] = pd.to_numeric(players["form"], errors="coerce").fillna(0)
            avg_form = players["form"].mean() if players["form"].mean() > 0 else 1
            form_multiplier = (players["form"] / avg_form).clip(lower=0.8, upper=1.2)
        else:
            form_multiplier = 1.0

        # --- Selected by % multiplier ---
        if "selected_by_percent" in players.columns:
            players["selected_by_percent"] = pd.to_numeric(
                players["selected_by_percent"], errors="coerce"
            ).fillna(0)
            avg_sel = (
                players["selected_by_percent"].mean()
                if players["selected_by_percent"].mean() > 0
                else 1
            )
            selected_multiplier = (players["selected_by_percent"] / avg_sel).clip(
                lower=0.9, upper=1.1
            )
        else:
            selected_multiplier = 1.0

        # --- Final EP next 3 fixtures with adjustments ---
        players["ep_next_3gw"] = (
            players["ep_per90"]
            * (players["expected_minutes_next"] / 90)
            * minutes_multiplier
            * form_multiplier
            * selected_multiplier
        )

        # --- Value metric ---
        players["cost_m"] = players["now_cost"] / 10.0
        players["ep_value"] = players["ep_next_3gw"] / players["cost_m"].replace(0, np.nan)

        # --- Clean NaNs ---
        players["ep_next_3gw"] = players["ep_next_3gw"].fillna(0)
        players["ep_per90"] = players["ep_per90"].fillna(0)
        players["ep_value"] = players["ep_value"].fillna(0)

        return players

class FPLTeamOptimizer:
    def __init__(self, players_df):
        self.players_df = players_df.copy()

    def optimize_squad(self, budget=100.0, squad_size=15, team_limit=3):
        # Map position codes
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        self.players_df["position_code"] = self.players_df["element_type"].map(pos_map)
        prob = pulp.LpProblem("FPL_Squad_Optimize", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", self.players_df["id"].tolist(), lowBound=0, upBound=1, cat="Integer")
        ep_map = self.players_df.set_index("id")["ep_next_3gw"].to_dict()
        prob += pulp.lpSum([ep_map[i] * x[i] for i in x]), "Total_EP"
        prob += pulp.lpSum([x[i] for i in x]) == squad_size
        cost_map = self.players_df.set_index("id")["now_cost"].to_dict()
        prob += pulp.lpSum([cost_map[i] * x[i] for i in x]) <= budget * 10
        # Position constraints
        for code, name in pos_map.items():
            ids = self.players_df[self.players_df["element_type"] == code]["id"].tolist()
            minp, maxp = constants.FORMATION_RULES[name]
            prob += pulp.lpSum([x[i] for i in ids]) >= minp
            prob += pulp.lpSum([x[i] for i in ids]) <= maxp
        # Team constraint
        for team_id, group in self.players_df.groupby("team"):
            ids = group["id"].tolist()
            prob += pulp.lpSum([x[i] for i in ids]) <= team_limit
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        selected_ids = [i for i in x if pulp.value(x[i]) >= 0.5]
        squad = self.players_df[self.players_df["id"].isin(selected_ids)].copy()
        total_cost = squad["now_cost"].sum()
        if total_cost > budget * 10:
            print(f"FPL Engine |    WARNING: Squad cost {total_cost/10:.1f} exceeds budget {budget:.1f}!")
        else:
            print(f"FPL Engine |    Squad cost: {total_cost/10:.1f} (Budget: {budget:.1f})")
        return squad

    def get_starters_and_bench(self, squad):
        # Use FPL formation rules for starting XI
        formation_min = {"DEF": 3, "MID": 2, "FWD": 1}
        formation_max = {"DEF": 5, "MID": 5, "FWD": 3}
        # Pick best GK
        gks = squad[squad["position_code"] == "GK"].sort_values("ep_next_3gw", ascending=False)
        starters = []
        if not gks.empty:
            starters.append(gks.iloc[0])
        # Outfield players
        outfield = squad[squad["position_code"] != "GK"].sort_values("ep_next_3gw", ascending=False)
        # Greedy: start with top 10 outfield, then adjust to fit formation
        chosen_rest = outfield.head(10)
        # Count by position
        counts = chosen_rest["position_code"].value_counts().to_dict()
        # Adjust to fit min constraints
        for pos, minp in formation_min.items():
            have = counts.get(pos, 0)
            if have < minp:
                needed = minp - have
                candidates = outfield[~outfield["id"].isin(chosen_rest["id"])]
                candidates = candidates[candidates["position_code"] == pos]
                to_add = candidates.head(needed)
                # Remove lowest EP from other positions with surplus
                for _ in range(needed):
                    surplus = [p for p in formation_min if counts.get(p, 0) > formation_min[p]]
                    if surplus:
                        # Remove lowest EP from surplus position
                        surplus_pos = surplus[0]
                        idx_to_remove = chosen_rest[chosen_rest["position_code"] == surplus_pos].sort_values("ep_next_3gw").index[:1]
                        if len(idx_to_remove) > 0:
                            chosen_rest = chosen_rest.drop(idx_to_remove[0])
                    # Add candidate
                    if not to_add.empty:
                        chosen_rest = pd.concat([chosen_rest, to_add.head(1)])
                        to_add = to_add.iloc[1:]
        # If any position above max, trim
        for pos, maxp in formation_max.items():
            have = chosen_rest["position_code"].value_counts().get(pos, 0)
            if have > maxp:
                idx_to_remove = chosen_rest[chosen_rest["position_code"] == pos].sort_values("ep_next_3gw").index[:have-maxp]
                chosen_rest = chosen_rest.drop(idx_to_remove)
        # Finalize starters
        starters_df = pd.concat([pd.DataFrame(starters), chosen_rest]).reset_index(drop=True)
        # Only keep 11 starters
        starters_df = starters_df.head(11)
        # Bench: next 4 highest EP not in starters
        bench_df = squad[~squad["id"].isin(starters_df["id"])] .sort_values("ep_next_3gw", ascending=False).head(4).reset_index(drop=True)

        # Select columns for output
        output_cols = [
            "web_name",      # name
            "team",         # team
            "position_code",# position
            "now_cost",     # cost
            "ep_next_3gw",  # expected points
            "total_points", # total points till now
            "form",         # form
            "xG",           # xG
            "xA"            # xA
        ]
        starters_df = starters_df[output_cols].copy()
        bench_df = bench_df[output_cols].copy()
        return starters_df, bench_df

class FPLSquadManager:
    def __init__(self, fpl_data, xgxa_df=None):
        self.fpl_data = fpl_data
        self.xgxa_df = xgxa_df

    def build_team(self):
        # Data cleaning/merging
        data_mgr = FPLDataManager(self.fpl_data["players"], self.fpl_data.get("teams"), self.fpl_data.get("fixtures"))
        
        # xG/xA: merge provided DataFrame
        if self.xgxa_df is not None:
            data_mgr.merge_xgxa(self.xgxa_df)
        players_df = data_mgr.clean()

        # EP calculation
        odds_engine = FPLOddsEngine(players_df, self.fpl_data.get("fixtures"), self.fpl_data.get("teams"))
        players_ep = odds_engine.compute_expected_points(xgxa_df=self.xgxa_df)
        print(f"FPL Engine |    Computed expected points for {len(players_ep)} players.")

        # Optimization
        optimizer = FPLTeamOptimizer(players_ep)
        squad = optimizer.optimize_squad()
        print(f"FPL Engine |    Optimized squad with {len(squad)} players.")

        starters, bench = optimizer.get_starters_and_bench(squad)
        print(f"FPL Engine |    Selected {len(starters)} starters and {len(bench)} bench players.")
        return {"starters": starters, "bench": bench, "players": players_ep}

    def build_odds_engine(self):
        """Builds the odds engine by fetching and processing live FPL data."""

        xg_df = None
        if self.xg_csv:
            xg_df = pd.read_csv(self.xg_csv)
        else:
            xg_df = self.generate_mock_xg_xa(self.players_df, seed=self.mock_xg_seed)

        odds_df = None
        if self.odds_csv:
            odds_df = pd.read_csv(self.odds_csv)
        else:
            # No odds data provided, we can skip this step
            print("No odds CSV provided, skipping odds processing.")

        players_with_ep = self.compute_expected_points(xg_df=xg_df, odds_df=odds_df)

        return players_with_ep

class FPLTeamBuilder:
    """
    A class to build and optimize an FPL team based on expected points.
    This is a placeholder for future team-building logic.
    """

    def __init__(self, players_df, budget=1000, squad_size=15, team_limit=3):
        self.players_df = players_df
        self.budget = budget
        self.squad_size = squad_size
        self.team_limit = team_limit

    def optimize_squad(self):
        """
        Return optimized squad using integer programming.
        - budget: in FPL units (e.g., 1000 == 100.0m) -- matches bootstrap now_cost units
        - formation_rules: dict specifying min/max starters per position if you want to enforce.

        Basic FPL constraints we implement:
        - total players = 15
        - squad composition by position: 2 GK, 5 DEF, 5 MID, 3 FWD (standard)
        - max 3 players from same real-world team
        - starting XI formation constraint: allow any valid formation on pitch (we will pick 11 highest EP playable while respecting 1 GK)

        Returns dictionary with lists (squad, starting_xi, bench)
        """
        players = self.players_df.copy()

        # Minutes multiplier: scale from 0.5 (low minutes) to 1.0 (90 min starter)
        minutes_ratio = (players["minutes"] / players["appearances"].replace(0, 1)) / 90
        minutes_multiplier = minutes_ratio.clip(lower=0.5, upper=1.0)

        # Fixture multiplier penalty: easier fixtures boost, harder fixtures reduce
        # difficulty: 1 easiest → 1.1 multiplier, 3 → 1.0, 5 hardest → 0.85
        fixture_multiplier_adj = (6 - players["difficulty"].fillna(3)) / 5
        fixture_multiplier_adj = fixture_multiplier_adj.clip(lower=0.85, upper=1.1)

        # Adjust EP score for optimization
        players["adj_ep"] = (
            players["ep_next_fixture"] * minutes_multiplier * fixture_multiplier_adj
        )

        # Map FPL element_type id to position code via players_df.element_type (1 GK,2 DEF,3 MID,4 FWD)
        # Get unique positions from 'element_type' column
        # We'll use players['element_type'] number to enforce.
        players["position_code"] = players["element_type"]

        # Build solver
        prob = pulp.LpProblem("FPL_Squad_Optimize", pulp.LpMaximize)

        # Decision variables: x_i = 1 if player i is selected in 15-man squad
        x = pulp.LpVariable.dicts(
            "x", players["id"].tolist(), lowBound=0, upBound=1, cat="Integer"
        )

        # Objective: maximize total ep_next_fixture of selected squad (we can later select starting XI by ep)
        ep_map = players.set_index("id")["adj_ep"].to_dict()
        prob += pulp.lpSum([ep_map[i] * x[i] for i in x]), "Total_EP"

        # Total players constraint
        prob += pulp.lpSum([x[i] for i in x]) == self.squad_size

        # Budget constraint (now_cost units)
        cost_map = players.set_index("id")["now_cost"].to_dict()
        prob += pulp.lpSum([cost_map[i] * x[i] for i in x]) <= self.budget

        # Position constraints
        # mapping element_type to codes: 1 GK,2 DEF,3 MID,4 FWD
        pos_map = players.groupby("element_type")["id"].apply(list).to_dict()
        code_to_name = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        for code, ids in pos_map.items():
            name = code_to_name.get(code, str(code))
            if name in constants.FORMATION_RULES:
                minp, maxp = constants.FORMATION_RULES[name]
                prob += pulp.lpSum([x[i] for i in ids]) >= minp
                prob += pulp.lpSum([x[i] for i in ids]) <= maxp

        # Club constraint: max 'team_limit' players from any FPL team
        for team_id, group in players.groupby("team"):
            ids = group["id"].tolist()
            prob += pulp.lpSum([x[i] for i in ids]) <= self.team_limit

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        selected_ids = [i for i in x if pulp.value(x[i]) >= 0.5]
        squad = players[players["id"].isin(selected_ids)].copy()

        # Now pick starting XI: 11 players with formation rules: 1 GK + between 3-5 DEF, 2-5 MID, 1-3 FWD
        starters = []
        bench = []
        # pick best GK by ep
        gks = squad[squad["element_type"] == 1].sort_values(
            "ep_next_fixture", ascending=False
        )
        if gks.empty:
            raise ValueError(
                "No GK in squad — solver should have enforced GK constraint"
            )
        starters.append(gks.iloc[0])
        bench_gk = gks.iloc[1:] if len(gks) > 1 else pd.DataFrame()

        # pick rest by EP but enforce formation legality
        others = squad[squad["element_type"] != 1].sort_values(
            "ep_next_fixture", ascending=False
        )
        # We'll try heuristic: choose top 10 others by ep and ensure formation valid by trying combinations
        # Simpler: greedily pick until have 10 more players but then adjust to fit minimums
        rest = others.copy()
        chosen_rest = rest.head(10)
        # Count by position
        counts = chosen_rest["element_type"].value_counts().to_dict()
        # map allowed ranges for starting XI positions
        start_rules = {2: (3, 5), 3: (2, 5), 4: (1, 3)}
        # If any position below min, replace lowest ep from other positions with next available from that position
        for code, (minp, maxp) in start_rules.items():
            have = counts.get(code, 0)
            if have < minp:
                needed = minp - have
                # find candidates from rest not selected
                candidates = rest[
                    ~rest.index.isin(chosen_rest.index) & (rest["element_type"] == code)
                ]
                if not candidates.empty:
                    to_add = candidates.head(needed)
                    # remove lowest ep players from chosen_rest that are in positions that can spare
                    # find positions with surplus
                    for _ in range(needed):
                        surplus_pos = None
                        for c2, (min2, max2) in start_rules.items():
                            have2 = (
                                chosen_rest["element_type"].value_counts().get(c2, 0)
                            )
                            if have2 > min2:
                                surplus_pos = c2
                                break
                        if surplus_pos is None:
                            break
                        # remove lowest ep from surplus_pos
                        idx_to_remove = (
                            chosen_rest[chosen_rest["element_type"] == surplus_pos]
                            .nlargest(0, "ep_next_fixture")
                            .index
                        )
                        if len(idx_to_remove) == 0:
                            idx_to_remove = (
                                chosen_rest[chosen_rest["element_type"] == surplus_pos]
                                .sort_values("ep_next_fixture")
                                .index[:1]
                            )
                        if len(idx_to_remove) > 0:
                            chosen_rest = chosen_rest.drop(idx_to_remove[0])
                    # add candidate
                    chosen_rest = pd.concat([chosen_rest, to_add.head(needed)])
        # finalize starters
        starters_df = pd.concat([pd.DataFrame([starters[0]]), chosen_rest]).reset_index(
            drop=True
        )
        # bench: remaining squad not in starters; sort by ep
        bench_df = squad[~squad["id"].isin(starters_df["id"])].sort_values(
            "ep_next_fixture", ascending=False
        )

        # arrange bench ordering: sub1, sub2, sub3
        bench_df = bench_df.reset_index(drop=True)

        return {
            "squad": squad.sort_values(
                ["element_type", "ep_next_fixture"], ascending=[True, False]
            ).reset_index(drop=True),
            "starters": starters_df.reset_index(drop=True),
            "bench": bench_df.reset_index(drop=True),
        }

    def build_and_optimize(self):
        opt = self.optimize_squad()
        return {"opt": opt}

class FPLTransferManager:
    def __init__(self, players_df):
        self.players_df = players_df.copy()

    def make_transfer(self, current_team, bank=0):
        my_team_ids = [p["element"] for p in current_team]
        
        my_team = self.players_df[self.players_df["id"].isin(my_team_ids)]

        # --- Find best transfer ---
        squad_ids = set(my_team_ids)
        candidates = self.players_df[~self.players_df["id"].isin(squad_ids)]

        best_delta = -999
        best_transfer = None

        for _, out_row in my_team.iterrows():
            max_price = out_row["now_cost"] + bank
            candidates = self.players_df[
                (self.players_df["element_type"] == out_row["element_type"]) &
                (~self.players_df["id"].isin(my_team_ids)) &
                (self.players_df["now_cost"] <= max_price)
            ]

            for _, in_row in candidates.iterrows():
                delta = in_row["ep_next_3gw"] - out_row["ep_next_3gw"]
                if delta > best_delta:
                    best_delta = delta
                    best_transfer = (out_row, in_row)

        if best_transfer:
            out_row, in_row = best_transfer
            print(f"Transfers  |    Suggested Transfer:")
            print(f"Transfers  |    OUT: {out_row['web_name']} ({out_row['ep_next_3gw']:.2f} EP, £{out_row['now_cost']/10:.1f}m)")
            print(f"Transfers  |    IN : {in_row['web_name']} ({in_row['ep_next_3gw']:.2f} EP, £{in_row['now_cost']/10:.1f}m)")
            print(f"Transfers  |    Δ Expected Points: {best_delta:.2f}")
