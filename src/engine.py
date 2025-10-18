import pandas as pd
import numpy as np
import pulp
from rapidfuzz import process, fuzz
import constants
import logger


# ------------------ Expected Points Engine ------------------ #
class FPLOddsEngine:
    def __init__(self, players_df, fixtures_df=None, teams_df=None):
        self.players_df = players_df.copy()
        self.fixtures_df = fixtures_df.copy() if fixtures_df is not None else None
        self.teams_df = teams_df.copy() if teams_df is not None else None

    def compute_expected_points(self, xgxa_df=None, odds_df=None):
        players = self.players_df.copy()
        players = self._convert_numeric_columns(players)
        players = self._merge_xgxa(players, xgxa_df)
        players = self._merge_odds(players, odds_df)
        players = self._compute_offensive_ep(players)
        players = self._compute_bonus(players)
        players = self._apply_fixture_multiplier(players)
        players = self._apply_team_strength(players)
        players = self._apply_odds_multiplier(players)
        players = self._apply_minutes_form_selected(players)
        players = self._finalize_ep(players)
        return players

    # ------------------ Private Helper Functions ------------------ #
    def _convert_numeric_columns(self, players):
        numeric_cols = [
            "form",
            "ict_index",
            "now_cost",
            "minutes",
            "total_points",
            "xG",
            "xA",
            "appearances",
            "selected_by_percent",
        ]
        for col in numeric_cols:
            if col in players.columns:
                players[col] = pd.to_numeric(players[col], errors="coerce").fillna(0)
        return players

    def _merge_xgxa(self, players, xgxa_df):
        if xgxa_df is None:
            return players
        xgxa_df["norm_name"] = xgxa_df["web_name"].str.replace(" ", "").str.lower()
        xgxa_names = xgxa_df["norm_name"].tolist()

        def fetch_xgxa(row):
            norm_name = (
                (str(row.get("first_name", "")) + str(row.get("second_name", "")))
                .replace(" ", "")
                .lower()
            )
            match, score, idx = process.extractOne(
                norm_name, xgxa_names, scorer=fuzz.ratio
            )
            if score >= 80:
                stats = xgxa_df.iloc[idx]
                return pd.Series([float(stats["xG"]), float(stats["xA"])])
            return pd.Series([0.0, 0.0])

        players[["xG", "xA"]] = players.apply(fetch_xgxa, axis=1)
        return players

    def _merge_odds(self, players, odds_df):
        if odds_df is None:
            return players
        key = (
            "id" if "id" in odds_df.columns and "id" in players.columns else "web_name"
        )
        players = players.merge(odds_df, on=key, how="left")
        return players

    def _compute_offensive_ep(self, players):
        p = constants.POINTS_PARAM
        players["minutes_factor"] = (
            np.log1p(players.get("minutes", 0)) * p["minutes_scaler"]
        )
        players["ep_offense_per90"] = (
            players["xG"] * p["goal_points"] + players["xA"] * p["assist_points"]
        )
        return players

    def _compute_bonus(self, players):
        p = constants.POINTS_PARAM
        if "ict_index" in players.columns:
            players["ict_index"] = pd.to_numeric(
                players["ict_index"], errors="coerce"
            ).fillna(10)
            players["bonus_prob"] = (
                players["ict_index"] / (players["ict_index"].max() + 1)
            ) * 0.8 + p["bonus_base"] * 0.2
        else:
            players["bonus_prob"] = np.clip(
                players["xG"] / (players["xG"].max() + 1) * 0.6 + p["bonus_base"], 0, 1
            )
        players["ep_bonus"] = players["bonus_prob"] * 3
        return players

    def _apply_fixture_multiplier(self, players):
        p = constants.POINTS_PARAM
        if self.fixtures_df is None or "kickoff_time" not in self.fixtures_df.columns:
            players["fixture_multiplier"] = 1.0
            return players

        self.fixtures_df["kickoff_time"] = pd.to_datetime(
            self.fixtures_df["kickoff_time"], errors="coerce"
        )
        now = pd.Timestamp.now(tz="UTC")
        future = self.fixtures_df[self.fixtures_df["kickoff_time"] >= now].sort_values(
            "kickoff_time"
        )

        home_diffs = (
            future.groupby("team_h")
            .head(3)[["team_h", "team_h_difficulty"]]
            .groupby("team_h")
            .mean()
            .reset_index()
            .rename(columns={"team_h": "team", "team_h_difficulty": "avg_difficulty"})
        )
        away_diffs = (
            future.groupby("team_a")
            .head(3)[["team_a", "team_a_difficulty"]]
            .groupby("team_a")
            .mean()
            .reset_index()
            .rename(columns={"team_a": "team", "team_a_difficulty": "avg_difficulty"})
        )
        avg_diffs = (
            pd.concat([home_diffs, away_diffs])
            .groupby("team")["avg_difficulty"]
            .mean()
            .reset_index()
        )

        players = players.merge(avg_diffs, on="team", how="left")
        players["fixture_multiplier"] = (
            (6 - players["avg_difficulty"].fillna(3)) / 5
        ).clip(
            lower=p.get("fixture_hard_penalty", 0.85),
            upper=p.get("fixture_easy_boost", 1.1),
        )
        return players

    def _apply_team_strength(self, players):
        if self.teams_df is None:
            players["team_strength"] = 1.0
            return players
        self.teams_df["strength_avg"] = (
            self.teams_df["strength_overall_home"]
            + self.teams_df["strength_overall_away"]
        ) / 2
        strength_map = self.teams_df.set_index("id")["strength_avg"].to_dict()
        players["team_strength"] = players["team"].map(strength_map).fillna(1.0)
        players["team_name"] = players["team"].map(
            self.teams_df.set_index("id")["name"]
        )
        players["team_strength"] = (
            players["team_strength"] / players["team_strength"].mean()
        )
        return players

    def _apply_odds_multiplier(self, players):
        p = constants.POINTS_PARAM
        if "goal_odds" in players.columns:
            players["goal_prob_from_odds"] = 1 / players["goal_odds"].replace(0, np.nan)
            players["odds_multiplier"] = 1 + (
                (
                    players["goal_prob_from_odds"].fillna(0)
                    - players["goal_prob_from_odds"].mean()
                )
                * p["bookmaker_weight"]
            )
        else:
            players["odds_multiplier"] = 1.0
        return players

    def _apply_minutes_form_selected(self, players):
        p = constants.POINTS_PARAM
        players["expected_minutes_next"] = np.where(
            players["minutes"] > 0,
            np.minimum(90, players["minutes"] / players.get("appearances", 1)),
            60,
        )
        minutes_ratio = (players["minutes"] / players.get("appearances", 1)) / 90
        players["minutes_multiplier"] = minutes_ratio.clip(
            lower=p.get("minutes_floor", 0.5), upper=1.0
        )

        if "form" in players.columns:
            avg_form = players["form"].mean() if players["form"].mean() > 0 else 1
            players["form_multiplier"] = (players["form"] / avg_form).clip(0.8, 1.2)
        else:
            players["form_multiplier"] = 1.0

        if "selected_by_percent" in players.columns:
            avg_sel = (
                players["selected_by_percent"].mean()
                if players["selected_by_percent"].mean() > 0
                else 1
            )
            players["selected_multiplier"] = (
                players["selected_by_percent"] / avg_sel
            ).clip(0.9, 1.1)
        else:
            players["selected_multiplier"] = 1.0
        return players

    def _finalize_ep(self, players):
        players["ep_per90"] = (
            (
                players["ep_offense_per90"]
                + players["ep_bonus"]
                + players["minutes_factor"]
            )
            * players["fixture_multiplier"]
            * players["odds_multiplier"]
            * players["team_strength"]
        )
        players["ep_next_3gw"] = (
            players["ep_per90"]
            * (players["expected_minutes_next"] / 90)
            * players["minutes_multiplier"]
            * players["form_multiplier"]
            * players["selected_multiplier"]
            * 3
        )
        players["cost_m"] = players["now_cost"] / 10
        players["ep_value"] = players["ep_next_3gw"] / players["cost_m"].replace(
            0, np.nan
        )
        players["ep_next_3gw"] = players["ep_next_3gw"].fillna(0)
        players["ep_per90"] = players["ep_per90"].fillna(0)
        players["ep_value"] = players["ep_value"].fillna(0)
        return players


# ------------------ Team Optimizer ------------------ #
class FPLTeamOptimizer:
    def __init__(self, players_df):
        self.players_df = players_df.copy()

    def optimize_squad(self, budget=100.0, squad_size=15, team_limit=3):
        self.players_df["position_code"] = self.players_df["element_type"].map(
            constants.POSITION_MAP
        )

        prob = pulp.LpProblem("FPL_Squad_Optimize", pulp.LpMaximize)
        x = pulp.LpVariable.dicts(
            "x", self.players_df["id"].tolist(), lowBound=0, upBound=1, cat="Integer"
        )

        ep_map = self.players_df.set_index("id")["ep_next_3gw"].to_dict()
        prob += pulp.lpSum([ep_map[i] * x[i] for i in x]), "Total_EP"
        prob += pulp.lpSum([x[i] for i in x]) == squad_size

        cost_map = self.players_df.set_index("id")["now_cost"].to_dict()
        prob += pulp.lpSum([cost_map[i] * x[i] for i in x]) <= budget * 10

        # Position & team constraints
        for code, name in constants.POSITION_MAP.items():
            ids = self.players_df[self.players_df["element_type"] == code][
                "id"
            ].tolist()
            minp, maxp = constants.FORMATION_RULES[name]
            prob += pulp.lpSum([x[i] for i in ids]) >= minp
            prob += pulp.lpSum([x[i] for i in ids]) <= maxp
        for team_id, group in self.players_df.groupby("team"):
            prob += pulp.lpSum([x[i] for i in group["id"].tolist()]) <= team_limit

        pulp.PULP_CBC_CMD(msg=False).solve(prob)
        selected_ids = [i for i in x if pulp.value(x[i]) >= 0.5]
        return self.players_df[self.players_df["id"].isin(selected_ids)].copy()


# ------------------ Transfer Manager ------------------ #
class FPLTransferManager:
    def __init__(self, players_df):
        self.players_df = players_df.copy()

    def _find_best_transfer(self, my_team, bank, max_transfers=1):
        transfers, remaining_bank = [], bank
        current_ep = my_team["ep_next_3gw"].sum()

        for t in range(max_transfers):
            best_delta, best_transfer = -999, None
            for _, out_row in my_team.iterrows():
                max_price = out_row["now_cost"] + remaining_bank

                # Count current players per real-life team
                team_counts = my_team["team"].value_counts().to_dict()

                # Candidate filtering
                candidates = self.players_df[
                    (self.players_df["element_type"] == out_row["element_type"])
                    & (~self.players_df["id"].isin(my_team["id"]))
                    & (self.players_df["now_cost"] <= max_price)
                ]

                # Enforce team limit
                candidates = candidates[
                    candidates["team"].apply(
                        lambda team_id: team_counts.get(team_id, 0)
                        < constants.TEAM_PLAYER_LIMIT
                        or team_id == out_row["team"]  # allow swapping within same team
                    )
                ]

                for _, in_row in candidates.iterrows():
                    delta = in_row["ep_next_3gw"] - out_row["ep_next_3gw"]
                    if delta > best_delta:
                        best_delta = delta
                        best_transfer = (out_row, in_row)

            if best_transfer:
                out_row, in_row = best_transfer
                my_team = my_team.drop(my_team[my_team["id"] == out_row["id"]].index)
                my_team = pd.concat(
                    [my_team, pd.DataFrame([in_row])], ignore_index=True
                )
                remaining_bank += out_row["now_cost"] - in_row["now_cost"]
                transfers.append(best_transfer)
            else:
                break

        if "position_code" not in my_team.columns and "element_type" in my_team.columns:
            my_team["position_code"] = my_team["element_type"].map(
                constants.POSITION_MAP
            )

        new_ep = my_team["ep_next_3gw"].sum()
        return my_team, transfers, current_ep, new_ep

    def make_multiple_transfers(self, current_team, num_transfers=1, bank=0):
        my_team_ids = [p["element"] for p in current_team]
        my_team = self.players_df[self.players_df["id"].isin(my_team_ids)].copy()
        new_team, transfers, current_ep, new_ep = self._find_best_transfer(
            my_team, bank, max_transfers=num_transfers
        )
        logger.log_transfer(transfers, current_ep, new_ep, double=False)
        return new_team
