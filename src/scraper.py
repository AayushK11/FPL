import os
import requests
import pandas as pd
from understatapi import UnderstatClient
import constants


class FPLDataFetcher:
    """Class to fetch FPL data from the API."""

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
        print(
            f"Scraper    |    Fetched {len(players)} players, {len(teams)} teams, and {len(positions)} positions."
        )
        fixtures = self.fetch_fixtures()
        print(f"Scraper    |    Fetched {len(fixtures)} fixtures.")
        return {
            "players": players,
            "teams": teams,
            "positions": positions,
            "fixtures": fixtures,
        }

    def get_xgxa_df(self, league, season):
        print(f"Scraper    |    Downloading xG/xA data for {league} {season}...")
        with UnderstatClient() as understat:
            data = understat.league(league=league).get_player_data(season=season)
        df = pd.DataFrame(data)[["id", "player_name", "xG", "xA"]]
        df.rename(columns={"player_name": "web_name"}, inplace=True)
        return df

    def build_xg_csv_season(self, league, season, output_file):
        with UnderstatClient() as understat:
            data = understat.league(league=league).get_player_data(season=season)
        df = pd.DataFrame(data)[["id", "player_name", "xG", "xA"]]
        df.rename(columns={"player_name": "web_name"}, inplace=True)
        df.to_csv(output_file, index=False)
        print(f"Saved xG/xA data to {output_file}")

    def fetch_user_team(self, entry_id: int, event_id: int):
        url = f"https://fantasy.premierleague.com/api/entry/{entry_id}/event/{event_id}/picks/"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["picks"], r.json()["entry_history"]["bank"] / 10.0
