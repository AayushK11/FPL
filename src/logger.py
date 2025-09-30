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


def format_columns(df):
    df["pos_order"] = df["position_code"].map(constants.REV_POSITION_MAP)
    df = df.sort_values(["pos_order", "ep_next_3gw"], ascending=[True, False]).drop(
        columns=["pos_order"]
    )
    return df[constants.OUTPUT_COLS]


def save_best_eleven(df):
    df = format_columns(df)
    save_csv(df, constants.BEST_ELEVEN_CSV_PATH, section="ENGINE")


def save_all_players(df):
    save_csv(df, constants.ALLPLAYERS_CSV_PATH, section="ENGINE")


def save_transfer_suggestions(df, team_name):
    file_name = constants.TRANSFER_SUGGESTION_CSV_PATH.format(teamname=team_name)
    df = format_columns(df)
    save_csv(df, file_name, section="TRANSFERS")
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
