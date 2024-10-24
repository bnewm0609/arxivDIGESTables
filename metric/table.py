from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Table:
    """Class for keeping track of table"""

    tabid: str
    schema: set[str]
    values: dict
    decontext_schema: Optional[dict[str, str]] = None
    dataframe: Optional[pd.DataFrame] = None
    caption: Optional[str] = None
