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
            logger.save_csv(self.players_df, constants.PLAYERS_RAW_PATH, "SCRAPER")

        if "fixtures" in self.fpl_data:
            logger.save_csv(
                self.fpl_data["fixtures"], constants.FIXTURES_RAW_PATH, "SCRAPER"
            )

        if "teams" in self.fpl_data:
            logger.save_csv(self.fpl_data["teams"], constants.TEAMS_RAW_PATH, "SCRAPER")

        self.xgxa_df = fetcher.get_xgxa_df("EPL", "2025")
        logger.save_csv(self.xgxa_df, constants.XGXA_CSV_PATH, "SCRAPER")

    # ------------------ Team Building ------------------ #
    def build_team(self):
        engine = FPLOddsEngine(
            self.players_df,
            fixtures_df=self.fpl_data.get("fixtures"),
            teams_df=self.fpl_data.get("teams"),
        )
        players_ep = engine.compute_expected_points(xgxa_df=self.xgxa_df)

        optimizer = FPLTeamOptimizer(players_ep)
        best_squad = optimizer.optimize_squad()

        self.result = {"players": players_ep, "best_eleven": best_squad}

    # ------------------ Output ------------------ #
    def output_results(self):
        if not self.result:
            logger.log("No team built yet.", "ENGINE")
            return

        # Save Best Eleven
        best_eleven = self.result["best_eleven"].copy()
        for col in constants.OUTPUT_COLS:
            if col not in best_eleven.columns:
                best_eleven[col] = 0.0 if col in constants.NUMERIC_COLUMNS else ""
        logger.save_best_eleven(best_eleven)

        # Save All Players
        all_players = self.result["players"].copy()
        for col in constants.OUTPUT_COLS:
            if col not in all_players.columns:
                all_players[col] = 0.0 if col in constants.NUMERIC_COLUMNS else ""
        logger.save_all_players(all_players)

    # ------------------ Transfer Suggestions ------------------ #
    def fetch_user_team(self, team_info, gw_id):
        fetcher = FPLDataFetcher()
        user_team, bank = fetcher.fetch_user_team(team_info["ENTRY_ID"], gw_id)
        logger.log(
            f"Fetched \"{team_info['NAME']}\" team owned by {team_info['MANAGER']} for GW {gw_id}.",
            "TRANSFERS",
        )
        return user_team, bank

    def suggest_transfers(self, team_info, gw_id):
        if not self.result:
            logger.log("No team built yet. Run build_team() first.", "TRANSFERS")
            return

        team, bank = self.fetch_user_team(team_info, gw_id)
        transfer_manager = FPLTransferManager(self.result["players"])

        if team_info["TRANSFER_LIMIT"] == 1:
            my_team = transfer_manager.make_single_transfer(team, bank)
        else:
            my_team = transfer_manager.make_double_transfer(team, bank)

        logger.save_transfer_suggestions(my_team, team_info["NAME"])

    def transfer_controller(self):
        for team in constants.TEAMS:
            self.suggest_transfers(team, constants.GW)

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
