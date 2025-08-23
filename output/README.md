# FPL Output Folder

This folder contains all generated outputs from the Fantasy Premier League (FPL) analysis and optimization scripts.

## Files

### 1. `all_players.csv`
- **Description:** Contains the full player dataset after computing expected points, multipliers, and other metrics.
- **Key Columns:**
  - `id` – Unique player ID.
  - `web_name` – Player name.
  - `team` – Team ID.
  - `element_type` – Player position code (1: GK, 2: DEF, 3: MID, 4: FWD).
  - `now_cost` – Current player cost (in 0.1m units).
  - `total_points` – Total FPL points accumulated.
  - `form` – Current player form.
  - `xG`, `xA` – Expected goals and assists.
  - `ep_per90` – Expected points per 90 minutes.
  - `ep_next_3gw` – Expected points for the next 3 gameweeks.
  - `ep_value` – Expected points per million.

### 2. `best11.csv`
- **Description:** Contains the top 11 players selected based on computed expected points for the next 3 gameweeks.
- **Columns:**
  - `web_name`, `team`, `position_code`, `now_cost`, `ep_next_3gw`, `total_points`, `form`, `xG`, `xA`.
- **Notes:** Sorted by position (`GK`, `DEF`, `MID`, `FWD`) and then by `ep_next_3gw` descending.

### 3. `*_gw*.csv`
- **Description:** Team-wise transfer suggestions and optimized squads for a given gameweek.
- **Filename Format:** `{teamname}_gw{gw}.csv`
- **Key Columns:**
  - `web_name`, `team`, `position_code`, `now_cost`, `ep_next_3gw`, `total_points`, `form`, `xG`, `xA`.
  - Includes suggested transfers (single or double) based on expected points.
- **Notes:** Automatically generated when running the transfer controller.

## Notes
- All CSVs are overwritten on each run to ensure they contain the latest expected points and transfer suggestions.ssw
- `best11.csv` is especially useful for quickly identifying the optimal starting lineup.
