from scraper import FPLDataFetcher
from engine import FPLOddsEngine, FPLTeamOptimizer, FPLTransferManager
from constants import *


class MainController:
    def __init__(self):
        self.fpl_data = {}
        self.players_df = None
        self.squad_manager = None
        self.result = None
        self.xgxa_df = None

    # ------------------ Data Fetching ------------------ #
    def fetch_data(self):
        fetcher = FPLDataFetcher()
        self.fpl_data = fetcher.fetch_live_fpl_data()

        if "players" in self.fpl_data:
            self.players_df = self.fpl_data["players"].copy()
            self.players_df.to_csv(PLAYERS_RAW_PATH, index=False)

        if "fixtures" in self.fpl_data:
            self.fpl_data["fixtures"].to_csv(FIXTURES_RAW_PATH, index=False)

        # Fetch xG/xA data
        self.xgxa_df = fetcher.get_xgxa_df("EPL", "2025")
        self.xgxa_df.to_csv(XGXA_CSV_PATH, index=False)

    # ------------------ Team Building ------------------ #
    def build_team(self):
        engine = FPLOddsEngine(
            self.players_df, fixtures_df=self.fpl_data.get("fixtures")
        )
        players_ep = engine.compute_expected_points(xgxa_df=self.xgxa_df)

        optimizer = FPLTeamOptimizer(players_ep)
        best_squad = optimizer.optimize_squad()

        # Split starters and bench
        self.result = {
            "players": players_ep,
            "starters": best_squad.iloc[:11].copy(),
            "bench": best_squad.iloc[11:].copy(),
        }

    # ------------------ Output ------------------ #
    def output_results(self):
        if not self.result:
            print("FPL Engine |    No team built yet.")
            return

        self.result["starters"].to_csv(STARTERS_CSV_PATH, index=False)
        print(f"FPL Engine |    Saved starters to {STARTERS_CSV_PATH}")

        self.result["bench"].to_csv(BENCH_CSV_PATH, index=False)
        print(f"FPL Engine |    Saved bench to {BENCH_CSV_PATH}")

        # Save all players
        all_players = self.result["players"].copy()
        output_cols = [
            "web_name",
            "team",
            "position_code",
            "now_cost",
            "ep_next_3gw",
            "total_points",
            "form",
            "xG",
            "xA",
        ]
        for col in output_cols:
            if col not in all_players.columns:
                all_players[col] = (
                    0.0
                    if col
                    in ["now_cost", "ep_next_3gw", "total_points", "form", "xG", "xA"]
                    else ""
                )
        all_players[output_cols].to_csv(ALLPLAYERS_CSV_PATH, index=False)
        print(f"FPL Engine |    Saved all players to {ALLPLAYERS_CSV_PATH}")

    # ------------------ Transfer Suggestions ------------------ #
    def fetch_user_team(self, entry_id, event_id):
        fetcher = FPLDataFetcher()
        user_team, bank = fetcher.fetch_user_team(entry_id, event_id)
        print(
            f"Transfers  |    Fetched user team for entry {entry_id} in event {event_id}."
        )
        return user_team, bank

    def suggest_transfers(self):
        if not self.result:
            print("Transfers |    No team built yet. Run build_team() first.")
            return

        team, bank = self.fetch_user_team(TEAM1.get("ENTRY_ID"), GW)
        transfer_manager = FPLTransferManager(self.result["players"])

        if TEAM1.get("TRANSFER_LIMIT", 1) == 1:
            my_team = transfer_manager.make_single_transfer(team, bank)
        else:
            my_team = transfer_manager.make_double_transfer(team, bank)

        output_cols = [
            "web_name",
            "team",
            "position_code",
            "now_cost",
            "ep_next_3gw",
            "total_points",
            "form",
            "xG",
            "xA",
        ]
        available_cols = [col for col in output_cols if col in my_team.columns]
        my_team[available_cols].to_csv(TRANSFER_SUGGESTION_CSV_PATH_TEAM1, index=False)
        print(
            f"Transfers  |    Saved transfer suggestions to {TRANSFER_SUGGESTION_CSV_PATH_TEAM1}"
        )

    # ------------------ Run Full Pipeline ------------------ #
    def run(self):
        self.fetch_data()
        print("Scraper    |    Data fetched and saved successfully.")
        print("-----------------------------------------------------------------------")

        self.build_team()
        print("FPL Engine |    Best XI Team built successfully.")
        print("-----------------------------------------------------------------------")

        self.output_results()
        print("FPL Engine |    Results output successfully.")
        print("-----------------------------------------------------------------------")

        self.suggest_transfers()
        print("Transfers  |    Suggested transfers based on current team.")
        print("-----------------------------------------------------------------------")
        print("FPL Engine |    All operations completed successfully.")


if __name__ == "__main__":
    MainController().run()
