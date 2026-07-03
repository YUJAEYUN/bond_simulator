"""주식 TR지수 + 채권 TR지수를 비중(equity_weight)과 리밸런싱 주기에 따라 혼합."""
from __future__ import annotations
import numpy as np
import pandas as pd


def align_series(equity_tr: pd.DataFrame, bond_tr: pd.DataFrame) -> pd.DataFrame:
    """
    두 TR 시계열(date, tr_index)을 날짜 교집합으로 정렬.
    같은 국가 페어(S&P500-미국채, KOSPI-한국채)이므로 거래일이 대체로 겹치고,
    소수 공휴일 차이는 교집합(inner join)으로 제거한다.
    """
    e = equity_tr[["date", "tr_index"]].rename(columns={"tr_index": "equity_tr"})
    b = bond_tr[["date", "tr_index"]].rename(columns={"tr_index": "bond_tr"})
    merged = pd.merge(e, b, on="date", how="inner").sort_values("date").reset_index(drop=True)
    merged["equity_ret"] = merged["equity_tr"].pct_change()
    merged["bond_ret"] = merged["bond_tr"].pct_change()
    return merged


def portfolio_tr(aligned: pd.DataFrame, equity_weight: float, rebalance_freq: str = "annual") -> pd.Series:
    """
    aligned: align_series() 결과 (date, equity_ret, bond_ret 포함)
    rebalance_freq: "none"(리밸런싱 없음, buy&hold) | "annual" | "quarterly"
    반환: 포트폴리오 TR 지수 시계열 (시작값 100), aligned와 같은 index
    """
    n = len(aligned)
    w_eq = equity_weight
    w_bd = 1.0 - equity_weight
    dates = aligned["date"].values
    eq_ret = aligned["equity_ret"].to_numpy()
    bd_ret = aligned["bond_ret"].to_numpy()

    if rebalance_freq == "none":
        rebalance_mask = np.zeros(n, dtype=bool)
    else:
        period = aligned["date"].dt.to_period("Y" if rebalance_freq == "annual" else "Q")
        rebalance_mask = (period != period.shift(1)).to_numpy().copy()
        rebalance_mask[0] = False

    port = np.empty(n)
    port[0] = 100.0
    eq_val = 100.0 * w_eq
    bd_val = 100.0 * w_bd

    for t in range(1, n):
        eq_val *= (1 + eq_ret[t]) if not np.isnan(eq_ret[t]) else 1.0
        bd_val *= (1 + bd_ret[t]) if not np.isnan(bd_ret[t]) else 1.0
        total = eq_val + bd_val
        if rebalance_mask[t] and total > 0:
            eq_val = total * w_eq
            bd_val = total * w_bd
        port[t] = eq_val + bd_val

    return pd.Series(port, index=aligned.index, name=f"w{equity_weight:.1f}_{rebalance_freq}")


def build_portfolio_grid(aligned: pd.DataFrame, weights: list[float], rebalance_freq: str = "annual") -> pd.DataFrame:
    """weights 각각에 대한 포트폴리오 TR 시계열을 컬럼으로 갖는 DataFrame (date 포함)."""
    out = pd.DataFrame({"date": aligned["date"].values})
    for w in weights:
        out[f"w{w:.2f}"] = portfolio_tr(aligned, w, rebalance_freq).values
    return out
