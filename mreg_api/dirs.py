"""Directory and file path constants."""

from __future__ import annotations

from pathlib import Path

import platformdirs

APP_NAME = "mreg-api"
APP_AUTHOR = "unioslo"

CONFIG_DIR = Path(platformdirs.user_config_dir(APP_NAME, APP_AUTHOR))
DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR))
LOG_DIR = Path(platformdirs.user_log_dir(APP_NAME, APP_AUTHOR))

# Config paths in order of precedence (highest first)
DEFAULT_CONFIG_PATH = [
    CONFIG_DIR / "mreg.conf",
    Path.home() / ".mreg-cli.conf",  # Legacy location
]

LOG_FILE_DEFAULT = LOG_DIR / "mreg-api.log"
HISTORY_FILE_DEFAULT = DATA_DIR / "history"
