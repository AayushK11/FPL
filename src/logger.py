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
