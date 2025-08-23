import requests
import pandas as pd
from understatapi import UnderstatClient
import constants
import logger


class FPLDataFetcher:
    """Fetch FPL and xG/xA data."""

    def __init__(self, timeout=60):
        self.timeout = timeout

    def fetch_players(self):
        r = requests.get(constants.FPL_API_URL, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        players = pd.DataFrame(j["elements"])
        teams = pd.DataFrame(j["teams"])
        positions = pd.DataFrame(j["element_types"])
        return players, teams, positions

    def fetch_fixtures(self):
        r2 = requests.get(constants.FIXTURES_API_URL, timeout=self.timeout)
        r2.raise_for_status()
        fixtures = pd.DataFrame(r2.json())
        return fixtures

    def fetch_live_fpl_data(self):
        players, teams, positions = self.fetch_players()
        logger.log(
            f"Fetched {len(players)} players, {len(teams)} teams, {len(positions)} positions.",
            "SCRAPER",
        )
        fixtures = self.fetch_fixtures()
        logger.log(f"Fetched {len(fixtures)} fixtures.", "SCRAPER")
        return {
            "players": players,
            "teams": teams,
            "positions": positions,
            "fixtures": fixtures,
        }

    def get_xgxa_df(self, league, season):
        logger.log(f"Downloading xG/xA data for {league} {season}...", "SCRAPER")
        with UnderstatClient() as understat:
            data = understat.league(league=league).get_player_data(season=season)
        df = pd.DataFrame(data)[["id", "player_name", "xG", "xA"]]
        df.rename(columns={"player_name": "web_name"}, inplace=True)
        return df

    def build_xg_csv_season(self, league, season, output_file):
        df = self.get_xgxa_df(league, season)
        df.to_csv(output_file, index=False)
        logger.log(f"Saved xG/xA data to {output_file}", "SCRAPER")

    def fetch_user_team(self, entry_id: int, event_id: int):
        url = constants.USER_TEAM_URL_TEMPLATE.format(entry_id=entry_id, event_id=event_id)
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["picks"], r.json()["entry_history"]["bank"] / 10.0
