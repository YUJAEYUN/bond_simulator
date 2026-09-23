"""주가지수 '변동 %' 시계열을 누적 지수(TR-equivalent, 시작값 100)로 변환."""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_equity_index(df: pd.DataFrame) -> pd.DataFrame:
    """df: date, close, change_pct (오름차순). 반환: date, tr_index (100 시작)."""
    r = df["change_pct"].to_numpy() / 100.0
    tr = np.empty(len(df))
    tr[0] = 100.0
    for t in range(1, len(df)):
        tr[t] = tr[t - 1] * (1 + r[t])
    return pd.DataFrame({"date": df["date"].values, "tr_index": tr})
