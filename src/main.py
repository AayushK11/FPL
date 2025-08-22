
from scraper import FPLDataFetcher
from engine import FPLSquadManager, FPLTransferManager
from constants import *

class MainController:
    def __init__(self):
        self.fpl_data = None
        self.squad_manager = None
        self.result = None

    def fetch_data(self):
        self.fpl_data = FPLDataFetcher().fetch_live_fpl_data()
        # Save raw/fetched data to data_fetched folder
        if "players" in self.fpl_data:
            self.fpl_data["players"].to_csv(PLAYERS_RAW_PATH, index=False)
        if "fixtures" in self.fpl_data:
            self.fpl_data["fixtures"].to_csv(FIXTURES_RAW_PATH, index=False)
        # Download xG/xA and save to CSV
        xgxa_df = FPLDataFetcher().get_xgxa_df("EPL", "2025")
        xgxa_df.to_csv("data_fetched/xg_xa.csv", index=False)
        self.xgxa_df = xgxa_df

    def build_team(self):
        self.squad_manager = FPLSquadManager(self.fpl_data, xgxa_df=self.xgxa_df)
        self.result = self.squad_manager.build_team()

    def output_results(self):
        self.result["starters"].to_csv(STARTERS_CSV_PATH, index=False)
        print(f"FPL Engine |    Saved starters to {STARTERS_CSV_PATH}")

        self.result["bench"].to_csv(BENCH_CSV_PATH, index=False)
        print(f"FPL Engine |    Saved bench to {BENCH_CSV_PATH}")

        # Dump all players output
        all_players = self.result["players"].copy()
        output_cols = [
            "web_name", "team", "position_code", "now_cost", "ep_next_3gw", "total_points", "form", "xG", "xA"
        ]
        # If columns missing, fill with 0 or blank
        for col in output_cols:
            if col not in all_players.columns:
                all_players[col] = 0.0 if col in ["now_cost", "ep_next_3gw", "total_points", "form", "xG", "xA"] else ""
        all_players[output_cols].to_csv(ALLPLAYERS_CSV_PATH, index=False)
        print(f"FPL Engine |    Saved all players to {ALLPLAYERS_CSV_PATH}")

    def fetch_user_team(self, entry_id, event_id):
        user_team, bank = FPLDataFetcher().fetch_user_team(entry_id, event_id)
        print(f"Transfers  |    Fetched user team for entry {entry_id} in event {event_id}:")
        return user_team, bank

    def suggest_transfers(self):
        if not self.result:
            print("Transfers |    No team built yet. Please run build_team() first.")
            return
        
        team, bank = self.fetch_user_team(TEAM1.get("ENTRY_ID"), GW)
        transfer_manager = FPLTransferManager(self.result["players"])
        if TEAM1.get("TRANSFER_LIMIT", 1) == 1:
            transfer_manager.make_single_transfer(team, bank)
        else:
            transfer_manager.make_double_transfer(team, bank)            
        
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

