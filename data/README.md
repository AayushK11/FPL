# FPL Data Folder

This folder contains all the raw and processed data used for Fantasy Premier League (FPL) analysis and squad optimization.

## Files

### 1. `players.csv`
- **Description:** Contains detailed player data fetched from the FPL API.

### 2. `fixtures.csv`
- **Description:** Contains upcoming fixtures for all Premier League teams.

### 3. `xgxa.csv`
- **Description:** Contains expected goals (xG) and expected assists (xA) data per player, fetched from Understat.

## Notes
- The CSVs are automatically refreshed on every run
- They are used in `engine.py` to calculate expected points and optimize squads.
- `xG` and `xA` values are merged with player data to enhance predictive accuracy.

