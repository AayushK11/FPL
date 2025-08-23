import constants

# ------------------ Log ------------------- #
def log(msg, section="INFO"):
    prefixes = {
        "SCRAPER": "[SCRAPER]",
        "ENGINE": "[ENGINE]",
        "TRANSFERS": "[TRANSFERS]",
        "INFO": "[INFO]",
    }
    prefix = prefixes.get(section, "[INFO]")
    print(f"{prefix} {msg}")


def separator():
    print("-" * 75)


# ------------------ CSV Output ------------------- #
def save_csv(df, path, section="INFO"):
    df_to_save = df.copy()

    if "position_code" in df_to_save.columns:
        df_to_save = df_to_save.sort_values("position_code")
    df_to_save.to_csv(path, index=False)
    log(f"Saved CSV to {path}", section)


# ------------------ Specialized CSV Helpers ------------------- #
def save_best_eleven(df):
    for col in constants.OUTPUT_COLS:
        if col not in df.columns:
            df[col] = (
                0.0
                if col
                in ["now_cost", "ep_next_3gw", "total_points", "form", "xG", "xA"]
                else ""
            )
    df = df[constants.OUTPUT_COLS]
    save_csv(df, constants.BEST_ELEVEN_CSV_PATH, section="ENGINE")


def save_all_players(df):
    save_csv(df, constants.ALLPLAYERS_CSV_PATH, section="ENGINE")


def save_transfer_suggestions(df, entry_id):
    csv_path = (
        constants.TRANSFER_SUGGESTION_CSV_PATH_TEAM1
        if entry_id == constants.TEAM1["ENTRY_ID"]
        else constants.TRANSFER_SUGGESTION_CSV_PATH_TEAM2
    )
    save_csv(df, csv_path, section="TRANSFERS")
    separator()


# ------------------ Transfer Logging ------------------- #
def log_transfer(transfers, current_ep, new_ep, double=False):
    if not transfers:
        log("No valid transfer(s) found.", "TRANSFERS")
        return

    header = "Suggested Double Transfer:" if double else "Suggested Transfer:"
    log(header, "TRANSFERS")

    for out_row, in_row in transfers:
        log(
            f"\tOUT: {out_row['web_name']} | EP: {out_row['ep_next_3gw']:.2f} | Cost: £{out_row['now_cost']/10:.1f}m",
            "TRANSFERS",
        )
        log(
            f"\tIN : {in_row['web_name']} | EP: {in_row['ep_next_3gw']:.2f} | Cost: £{in_row['now_cost']/10:.1f}m",
            "TRANSFERS",
        )

    log(
        f"Current XI EP: {current_ep:.2f} | New XI EP: {new_ep:.2f} | Δ EP: {new_ep-current_ep:.2f}",
        "TRANSFERS",
    )
