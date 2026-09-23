"""H2: 주식-채권 음의 상관관계(하락장 방어) 가설 검증."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def full_period_correlation(aligned: pd.DataFrame) -> dict:
    d = aligned.dropna(subset=["equity_ret", "bond_ret"])
    r, p = stats.pearsonr(d["equity_ret"], d["bond_ret"])
    return {"n": len(d), "pearson_r": r, "p_value": p}


def crisis_conditional_correlation(aligned: pd.DataFrame, crisis_percentile: float = 0.05) -> dict:
    """주식 일별수익률 하위 percentile 구간(=하락일)에서의 상관계수."""
    d = aligned.dropna(subset=["equity_ret", "bond_ret"])
    cutoff = d["equity_ret"].quantile(crisis_percentile)
    crisis = d[d["equity_ret"] <= cutoff]
    r, p = stats.pearsonr(crisis["equity_ret"], crisis["bond_ret"])
    return {
        "n": len(crisis), "cutoff_return_pct": cutoff * 100,
        "pearson_r": r, "p_value": p,
        "mean_bond_ret_on_crisis_days_pct": crisis["bond_ret"].mean() * 100,
    }


def rolling_correlation(aligned: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    d = aligned.dropna(subset=["equity_ret", "bond_ret"]).reset_index(drop=True)
    roll = d["equity_ret"].rolling(window).corr(d["bond_ret"])
    return pd.DataFrame({"date": d["date"], "rolling_corr": roll})


def yearly_correlation(aligned: pd.DataFrame) -> pd.DataFrame:
    d = aligned.dropna(subset=["equity_ret", "bond_ret"]).copy()
    d["year"] = d["date"].dt.year
    rows = []
    for y, g in d.groupby("year"):
        if len(g) < 20:
            continue
        r, p = stats.pearsonr(g["equity_ret"], g["bond_ret"])
        rows.append({"year": y, "n": len(g), "pearson_r": r, "p_value": p})
    return pd.DataFrame(rows)


def scatter_data(aligned: pd.DataFrame) -> pd.DataFrame:
    d = aligned.dropna(subset=["equity_ret", "bond_ret"])
    return d[["date", "equity_ret", "bond_ret"]].copy()
