"""
investing.com 스타일 CSV(날짜, 종가, 시가, 고가, 저가, [거래량], 변동 %) 로더.

주가지수(KOSPI, S&P500) CSV의 '종가'는 지수 포인트이고,
채권 CSV의 '종가'는 만기수익률(%, 예: 4.464 = 4.464%)이다.
두 경우 모두 '변동 %' 컬럼은 전일 대비 종가의 변화율이며,
기존 Best-Days 시뮬레이터(JS)와 동일하게 이 컬럼을 일별 수익률의 원천으로 사용한다.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s.replace(" ", ""), format="%Y-%m-%d")


def load_raw_csv(path: str) -> pd.DataFrame:
    """단일 CSV 파일을 읽어 date/close/change_pct 컬럼의 DataFrame으로 반환 (오름차순 정렬)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    date = df["날짜"].astype(str).str.replace(" ", "", regex=False)
    date = pd.to_datetime(date, format="%Y-%m-%d")

    close = (
        df["종가"].astype(str).str.replace(",", "", regex=False).str.strip().astype(float)
    )

    change = (
        df["변동 %"].astype(str).str.replace("%", "", regex=False).str.replace(" ", "", regex=False)
    )
    change = pd.to_numeric(change, errors="coerce")

    out = pd.DataFrame({"date": date, "close": close, "change_pct": change})
    out = out.dropna(subset=["date", "close", "change_pct"])
    out = out.sort_values("date").drop_duplicates(subset="date", keep="last")
    out = out.reset_index(drop=True)
    return out


def load_series(paths: list[str]) -> pd.DataFrame:
    """여러 파일(기간별로 나뉜 동일 자산 데이터)을 이어붙여 하나의 시계열로 반환."""
    parts = [load_raw_csv(p) for p in paths]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates(subset="date", keep="last")
    combined = combined.reset_index(drop=True)
    return combined


if __name__ == "__main__":
    import sys
    df = load_series(sys.argv[1:])
    print(df.head())
    print(df.tail())
    print(len(df), "rows", df["date"].min(), "~", df["date"].max())
