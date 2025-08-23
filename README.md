# Fantasy Premier League Optimizer

This project calculates expected points (EP) for FPL players, optimizes squads, and suggests transfers for multiple teams.

---

## Project Structure

```
FPL_Project/
│
├── data/                     # Raw and cached data
│   ├── players.csv           # Player stats
│   ├── fixtures.csv          # Fixture info
│   └── xg_xa.csv             # Understat xG/xA data
│
├── output/                   # Generated outputs
│   ├── all_players.csv       # Full players dataset with EP
│   ├── best11.csv            # Best 11 lineup based on EP
│   └── *_gw*.csv             # Team-wise transfer suggestions & optimized squads
│
├── engine.py                 # Core engine: EP calculation, team optimization, transfers
├── main.py                   # Main script to run the project
├── constants.py              # Constants: positions, scoring rules, teams, paths
├── logger.py                 # Logging and saving outputs
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation
```

---

## Data Folder

All raw and fetched data is stored here:

- `players.csv` – Player stats from FPL.
- `fixtures.csv` – Fixture information for upcoming gameweeks.
- `xg_xa.csv` – Expected goals and assists data from Understat.

---

## Output Folder

All outputs and generated files are stored here:

- `all_players.csv` – Full player dataset with calculated EP.
- `best11.csv` – Best 11 lineup based on expected points.
- `*_gw*.csv` – Team-wise optimized squads and transfer suggestions per gameweek.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/AayushK11/FPL.git
cd FPL
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\\Scripts\\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r pip_requirements.txt
```

---

## Usage

Run the main script:

```bash
python main.py
```

- The script calculates expected points, optimizes squads, and saves outputs.
- Transfer suggestions for all teams defined in `constants.py` are also saved automatically.

---

## Constants

- Gameweek (`GW`) and teams (`TEAM1`, `TEAM2`, etc.) are defined in `constants.py`.
- Add more teams in `constants.py` to automatically include them in transfers.

---

## Logging

- Logs and transfer suggestions are saved using `logger.py`.
- Team-wise suggestions are saved in `output/{teamname}_{gw}.csv`.

---

## Dependencies

- pandas  
- numpy  
- pulp  
- RapidFuzz  
- requests  
- understatapi  

Check `pip_requirements.txt` for full versions.

---

## License

MIT License
