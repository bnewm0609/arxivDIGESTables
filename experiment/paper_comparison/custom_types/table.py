from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Table:
    """Class for keeping track of table"""
    tabid: str
    schema: set[str]
    values: dict
    id: Optional[int] = None
    type: Optional[str] = None
    dataframe: Optional[pd.DataFrame] = None
    caption: Optional[str] = None
    icscaption: Optional[dict] = None
    icspaper: Optional[dict] = None
    intextref: Optional[dict] = None
