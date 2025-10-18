import constants
import pandas as pd
from typing import Dict, Any, Optional, List


# ------------------ Exported Functions ------------------- #
def log(msg: str, section: str = "INFO") -> None:
    """Wrapper function for Logger.log for backward compatibility"""
    Logger.log(msg, section)


def separator() -> None:
    """Wrapper function for Logger.separator for backward compatibility"""
    Logger.separator()


def save_csv(df: pd.DataFrame, path: str, section: str = "INFO") -> None:
    """Wrapper function for CSVFormatter and saving functionality"""
    save_formatted_csv(df, path, section)


# ------------------ Core Classes ------------------- #
class ColumnConfig:
    """Configuration for column formatting"""

    def __init__(self, width: int, align: str = "left", header: str = None):
        self.width = width
        self.align = align
        self.header = header
        self.format_type: Optional[str] = None
        self.precision: Optional[int] = None
        self.divide_by_ten: bool = False

    @classmethod
    def numeric(
        cls,
        width: int,
        precision: int = None,
        divide_by_ten: bool = False,
        align: str = "right",
        header: str = None,
    ) -> "ColumnConfig":
        config = cls(width, align, header)
        config.format_type = "numeric"
        config.precision = precision
        config.divide_by_ten = divide_by_ten
        return config

    @classmethod
    def text(
        cls, width: int, align: str = "left", header: str = None
    ) -> "ColumnConfig":
        config = cls(width, align, header)
        config.format_type = "text"
        return config


class Formatter:
    """Handles text and number formatting"""

    @staticmethod
    def format_number(value: Any, config: ColumnConfig) -> str:
        try:
            number = float(value)
            if config.divide_by_ten:
                number /= 10
            if config.precision is not None:
                result = f"{number:.{config.precision}f}"
            else:
                result = str(int(number))
            return result.rjust(config.width)
        except (ValueError, TypeError):
            return str(value).rjust(config.width)

    @staticmethod
    def format_text(value: Any, config: ColumnConfig) -> str:
        text = str(value)
        if config.align == "left":
            return text.ljust(config.width)[: config.width]
        elif config.align == "right":
            return text.rjust(config.width)[: config.width]
        else:  # center
            return text.center(config.width)[: config.width]

    @classmethod
    def format_value(cls, value: Any, config: ColumnConfig) -> str:
        if config.format_type == "numeric":
            return cls.format_number(value, config)
        return cls.format_text(value, config)


class CSVFormatter:
    """Handles CSV file formatting and saving"""

    COLUMN_CONFIGS = {
        "web_name": ColumnConfig.text(20, header="Player Name"),
        "team_name": ColumnConfig.text(15, header="Team"),
        "position_code": ColumnConfig.text(3, "center", "Pos"),
        "now_cost": ColumnConfig.numeric(6, 1, True, header="Price"),
        "ep_next_3gw": ColumnConfig.numeric(7, 2, header="EP"),
        "total_points": ColumnConfig.numeric(4, header="Pts"),
        "form": ColumnConfig.numeric(5, 1, header="Form"),
        "xG": ColumnConfig.numeric(5, 2, header="xG"),
        "xA": ColumnConfig.numeric(5, 2, header="xA"),
    }

    @classmethod
    def format_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Format the dataframe according to column configurations"""
        df_formatted = df.copy()

        for col, config in cls.COLUMN_CONFIGS.items():
            if col in df_formatted.columns:
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: Formatter.format_value(x, config)
                )
                # Update column name to formatted header
                if config.header:
                    formatted_header = Formatter.format_value(config.header, config)
                    df_formatted = df_formatted.rename(columns={col: formatted_header})

        return df_formatted


class Logger:
    """Handles logging and output operations"""

    PREFIXES = {
        "SCRAPER": "[SCRAPER]",
        "ENGINE": "[ENGINE]",
        "TRANSFERS": "[TRANSFERS]",
        "INFO": "[INFO]",
    }

    @classmethod
    def log(cls, msg: str, section: str = "INFO") -> None:
        prefix = cls.PREFIXES.get(section, "[INFO]")
        print(f"{prefix} {msg}")

    @classmethod
    def separator(cls) -> None:
        print("-" * 75)

    @classmethod
    def log_transfer(
        cls,
        transfers: List[tuple],
        current_ep: float,
        new_ep: float,
        double: bool = False,
    ) -> None:
        """Log transfer suggestions with detailed information"""
        if not transfers:
            cls.log("No valid transfer(s) found.", "TRANSFERS")
            return

        header = "Suggested Double Transfer:" if double else "Suggested Transfer:"
        cls.log(header, "TRANSFERS")

        for out_row, in_row in transfers:
            cls.log(
                f"\tOUT: {out_row['web_name']} | "
                f"EP: {out_row['ep_next_3gw']:.2f} | "
                f"Cost: £{out_row['now_cost']/10:.1f}m",
                "TRANSFERS",
            )
            cls.log(
                f"\tIN : {in_row['web_name']} | "
                f"EP: {in_row['ep_next_3gw']:.2f} | "
                f"Cost: £{in_row['now_cost']/10:.1f}m",
                "TRANSFERS",
            )

        cls.log(
            f"Current XI EP: {current_ep:.2f} | "
            f"New XI EP: {new_ep:.2f} | "
            f"Δ EP: {new_ep-current_ep:.2f}",
            "TRANSFERS",
        )


# ------------------ File Operations ------------------- #
def save_formatted_csv(df: pd.DataFrame, path: str, section: str = "INFO") -> None:
    """Save formatted dataframe to CSV"""
    formatted_df = CSVFormatter.format_dataframe(df)
    formatted_df.to_csv(path, index=False)
    Logger.log(f"Saved CSV to {path}", section)


def format_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Format and sort columns for output"""
    df = df.copy()
    df["pos_order"] = df["position_code"].map(constants.REV_POSITION_MAP)
    df = df.sort_values(["pos_order", "ep_next_3gw"], ascending=[True, False])
    df = df[constants.OUTPUT_COLS]
    if "pos_order" in df.columns:
        df = df.drop(columns=["pos_order"])
    return df


def save_best_eleven(df: pd.DataFrame) -> None:
    """Save best eleven players"""
    formatted_df = format_columns(df)
    save_formatted_csv(formatted_df, constants.BEST_ELEVEN_CSV_PATH, section="ENGINE")


def save_all_players(df: pd.DataFrame) -> None:
    """Save all players"""
    save_formatted_csv(df, constants.ALLPLAYERS_CSV_PATH, section="ENGINE")


def save_transfer_suggestions(df: pd.DataFrame, team_name: str) -> None:
    """Save transfer suggestions"""
    file_name = constants.TRANSFER_SUGGESTION_CSV_PATH.format(teamname=team_name)
    formatted_df = format_columns(df)
    save_formatted_csv(formatted_df, file_name, section="TRANSFERS")
    Logger.separator()


def log_transfer(
    transfers: List[tuple], current_ep: float, new_ep: float, double: bool = False
) -> None:
    """Log transfer suggestions with detailed information"""
    Logger.log_transfer(transfers, current_ep, new_ep, double=double)
