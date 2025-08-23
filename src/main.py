from scraper import FPLDataFetcher
from engine import FPLOddsEngine, FPLTeamOptimizer, FPLTransferManager
import constants
import logger


class MainController:
    def __init__(self):
        self.fpl_data = {}
        self.players_df = None
        self.result = None
        self.xgxa_df = None

    # ------------------ Data Fetching ------------------ #
    def fetch_data(self):
        fetcher = FPLDataFetcher()
        self.fpl_data = fetcher.fetch_live_fpl_data()

        if "players" in self.fpl_data:
            self.players_df = self.fpl_data["players"].copy()
            self.players_df.to_csv(constants.PLAYERS_RAW_PATH, index=False)

        if "fixtures" in self.fpl_data:
            self.fpl_data["fixtures"].to_csv(constants.FIXTURES_RAW_PATH, index=False)

        self.xgxa_df = fetcher.get_xgxa_df("EPL", "2025")
        self.xgxa_df.to_csv(constants.XGXA_CSV_PATH, index=False)

    # ------------------ Team Building ------------------ #
    def build_team(self):
        engine = FPLOddsEngine(
            self.players_df, fixtures_df=self.fpl_data.get("fixtures")
        )
        players_ep = engine.compute_expected_points(xgxa_df=self.xgxa_df)

        optimizer = FPLTeamOptimizer(players_ep)
        best_squad = optimizer.optimize_squad()

        self.result = {"players": players_ep, "best_eleven": best_squad}

    # ------------------ Output ------------------ #
    def output_results(self):
        if not self.result:
            logger.log("No team built yet", "ENGINE")
            return

        best_eleven = self.result["best_eleven"].copy()
        all_players = self.result["players"].copy()

        for col in constants.OUTPUT_COLS:
            if col not in best_eleven.columns:
                best_eleven[col] = (
                    0.0
                    if col
                    in ["now_cost", "ep_next_3gw", "total_points", "form", "xG", "xA"]
                    else ""
                )

        best_eleven[constants.OUTPUT_COLS].to_csv(
            constants.BEST_ELEVEN_CSV_PATH, index=False
        )

        logger.log(f"Saved best eleven to {constants.BEST_ELEVEN_CSV_PATH}", "ENGINE")

        for col in constants.OUTPUT_COLS:
            if col not in all_players.columns:
                all_players[col] = (
                    0.0
                    if col
                    in ["now_cost", "ep_next_3gw", "total_points", "form", "xG", "xA"]
                    else ""
                )

        all_players[constants.OUTPUT_COLS].to_csv(
            constants.ALLPLAYERS_CSV_PATH, index=False
        )

        logger.log(f"Saved all players to {constants.ALLPLAYERS_CSV_PATH}", "ENGINE")

    # ------------------ Transfer Suggestions ------------------ #
    def fetch_user_team(self, entry_id, event_id):
        fetcher = FPLDataFetcher()
        user_team, bank = fetcher.fetch_user_team(entry_id, event_id)

        logger.log(
            f"Fetched user team for entry {entry_id} in event {event_id}.", "TRANSFERS"
        )
        return user_team, bank

    def suggest_transfers(self, entry_id, transfer_limit, event_id):
        if not self.result:
            logger.log("No team built yet. Run build_team() first.", "TRANSFERS")
            return

        team, bank = self.fetch_user_team(entry_id, event_id)
        transfer_manager = FPLTransferManager(self.result["players"])

        if transfer_limit == 1:
            my_team = transfer_manager.make_single_transfer(team, bank)
        else:
            my_team = transfer_manager.make_double_transfer(team, bank)

        csv_path = (
            constants.TRANSFER_SUGGESTION_CSV_PATH_TEAM1
            if entry_id == constants.TEAM1["ENTRY_ID"]
            else constants.TRANSFER_SUGGESTION_CSV_PATH_TEAM2
        )

        my_team.to_csv(csv_path, index=False)
        logger.log(f"Saved transfer suggestions to {csv_path}", "TRANSFERS")
        logger.separator()

    def transfer_controller(self):
        self.suggest_transfers(
            constants.TEAM1["ENTRY_ID"], constants.TEAM1["TRANSFER_LIMIT"], constants.GW
        )
        self.suggest_transfers(
            constants.TEAM2["ENTRY_ID"], constants.TEAM2["TRANSFER_LIMIT"], constants.GW
        )

    # ------------------ Run Full Pipeline ------------------ #
    def run(self):
        self.fetch_data()
        logger.log("Data fetched and saved successfully.", "SCRAPER")
        logger.separator()

        self.build_team()
        logger.log("Best XI Team built successfully.", "ENGINE")
        logger.separator()

        self.output_results()
        logger.log("Results output successfully.", "ENGINE")
        logger.separator()

        self.transfer_controller()
        logger.log("Transfer suggestions generated successfully.", "TRANSFERS")
        logger.separator()


if __name__ == "__main__":
    MainController().run()
