"""
채권 총수익(Total Return)지수 재구성 모듈.

## 배경
보유한 채권 데이터는 ETF/펀드의 가격+분배금이 아니라 '만기수익률(YTM, %)' 시계열이다
(미국/한국 10년 만기 국채, constant-maturity yield). 따라서 스펙 2.2의
"가격 + 분배금 chain-linking" 방식을 그대로 쓸 수 없고, 스펙 2.5·섹션6에서 명시한
대체 경로("국고채 금리 원자료로 대체 재구성")를 따른다.

## 방법론: Constant-Maturity 가상 액면채권(par bond) 재가격 모델
매일 "잔존만기 T년짜리 액면채권(가격=100, 표면금리=어제 종가 수익률)"을 가정하고,
- 하루 뒤 그 채권을 오늘자 수익률로 재할인해 가격 변화(자본손익)를 구하고
- 경과일수만큼 표면이자를 일할 누적(캐리 수익)한다
매일 다시 par(가격 100)로 "롤"하여 다음날 새 표면금리로 반복한다. 이는 듀레이션·컨벡시티를
선형 근사가 아니라 실제 채권가격공식으로 반영하는 방식이며, CMT(Constant Maturity Treasury)
총수익지수를 근사할 때 흔히 쓰이는 기법이다.

## 한계 (명시)
- 연 1회 이표 지급을 가정한 단순화된 채권가격 공식 사용 (실제 미 국채는 반기 이표).
  방향성·상대적 크기 비교에는 무리가 없으나 절대 수익률 값은 근사치임.
- 잔존만기는 항상 T년으로 고정(실제로는 매일 살짝 감소) — CMT 지수 정의와 일치.
- 세전 기준, 신용/유동성 스프레드 없음 (국채 자체이므로 스프레드 이슈는 없음).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _par_bond_price(yield_dec: np.ndarray, coupon_dec: np.ndarray, maturity_years: float) -> np.ndarray:
    """연 1회 이표, 만기 T년, 액면 100인 채권의 가격을 수익률 yield_dec로 할인한 값."""
    T = maturity_years
    y = yield_dec
    c = coupon_dec * 100.0

    with np.errstate(divide="ignore", invalid="ignore"):
        annuity_factor = np.where(
            np.abs(y) > 1e-8,
            (1 - (1 + y) ** (-T)) / np.where(np.abs(y) > 1e-8, y, 1.0),
            T,  # y -> 0 극한
        )
    price = c * annuity_factor + 100.0 * (1 + y) ** (-T)
    return price


def build_bond_total_return(yield_df: pd.DataFrame, maturity_years: float = 10.0) -> pd.DataFrame:
    """
    yield_df: columns [date, close(=연 수익률 %), change_pct] (오름차순 정렬)
    반환: date, yield_pct, daily_return, tr_index (시작값 100)
    """
    df = yield_df.sort_values("date").reset_index(drop=True).copy()
    y = df["close"].to_numpy() / 100.0  # % -> decimal
    dates = df["date"].to_numpy()

    n = len(df)
    daily_return = np.zeros(n)
    daily_return[:] = np.nan

    # 경과 일수 (달력일 기준, 주말/휴일 이자도 누적)
    day_gap = np.zeros(n)
    day_gap[1:] = (df["date"].values[1:] - df["date"].values[:-1]) / np.timedelta64(1, "D")

    for t in range(1, n):
        coupon_prev = y[t - 1]
        price_today = _par_bond_price(np.array([y[t]]), np.array([coupon_prev]), maturity_years)[0]
        accrued = coupon_prev * 100.0 * (day_gap[t] / 365.0)
        daily_return[t] = (price_today + accrued) / 100.0 - 1.0

    tr_index = np.empty(n)
    tr_index[0] = 100.0
    for t in range(1, n):
        tr_index[t] = tr_index[t - 1] * (1 + daily_return[t])

    out = pd.DataFrame({
        "date": df["date"].values,
        "yield_pct": df["close"].values,
        "daily_return": daily_return,
        "tr_index": tr_index,
    })
    return out


if __name__ == "__main__":
    import sys
    from data_loader import load_raw_csv
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/us10y_yield.csv"
    ydf = pd.read_csv(path, parse_dates=["date"])
    tr = build_bond_total_return(ydf)
    print(tr.head(10))
    print(tr.tail(10))
    years = (tr["date"].iloc[-1] - tr["date"].iloc[0]).days / 365.25
    cagr = (tr["tr_index"].iloc[-1] / tr["tr_index"].iloc[0]) ** (1 / years) - 1
    print(f"CAGR: {cagr*100:.2f}%  over {years:.1f}y")
