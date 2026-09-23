"""
기존 index.html의 Best-Days 제거 로직(JS)을 Python으로 이식.
상위 수익일 N개를 0%로 교체했을 때 연평균 수익률(CAGR) 변화를 계산한다.
확장: 원지수(가격 기준)뿐 아니라 혼합 포트폴리오 TR 시계열에도 동일 로직을 적용해,
채권을 섞으면 "몇 안 되는 날에 의존하는 정도"가 줄어드는지 부가 확인한다 (스펙 3.4, 우선순위 낮음).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def calc_cagr_after_removal(dates: np.ndarray, daily_returns: np.ndarray, n: int) -> dict:
    """
    dates: 오름차순 정렬된 날짜 배열
    daily_returns: 각 시점의 일별 수익률(decimal, 첫 값은 NaN 가능)
    n: 제거할 상위 수익일 개수
    """
    valid = ~np.isnan(daily_returns)
    r = daily_returns[valid]
    d = dates[valid]

    order = np.argsort(-r)  # 내림차순
    top_idx = order[:n] if n > 0 else np.array([], dtype=int)
    mask = np.ones(len(r), dtype=bool)
    mask[top_idx] = False

    v_orig = np.prod(1 + r)
    v_mod = np.prod(1 + np.where(mask, r, 0.0))

    years = (d[-1] - d[0]) / np.timedelta64(1, "D") / 365.25
    orig_cagr = v_orig ** (1 / years) - 1
    mod_cagr = v_mod ** (1 / years) - 1

    return {
        "n_removed": n, "years": years,
        "orig_cagr": orig_cagr, "mod_cagr": mod_cagr,
        "cagr_diff_pp": (mod_cagr - orig_cagr) * 100,
        "top_days": [(pd.Timestamp(d[i]), r[i]) for i in top_idx],
    }


def best_days_table(dates: np.ndarray, daily_returns: np.ndarray, ns: list[int]) -> pd.DataFrame:
    rows = []
    for n in ns:
        res = calc_cagr_after_removal(dates, daily_returns, n)
        rows.append({
            "n_removed": n, "orig_cagr_pct": res["orig_cagr"] * 100,
            "mod_cagr_pct": res["mod_cagr"] * 100, "diff_pp": res["cagr_diff_pp"],
        })
    return pd.DataFrame(rows)


def best_days_by_weight(aligned: pd.DataFrame, portfolio_grid: pd.DataFrame, weights: list[float], ns: list[int]) -> pd.DataFrame:
    """비중별 포트폴리오 TR 시계열에 best-days 제거 로직 적용."""
    dates = aligned["date"].values
    rows = []
    for w in weights:
        col = f"w{w:.2f}"
        tr = portfolio_grid[col].to_numpy()
        daily_ret = np.empty(len(tr))
        daily_ret[0] = np.nan
        daily_ret[1:] = tr[1:] / tr[:-1] - 1
        for n in ns:
            res = calc_cagr_after_removal(dates, daily_ret, n)
            rows.append({
                "weight": w, "n_removed": n,
                "orig_cagr_pct": res["orig_cagr"] * 100,
                "mod_cagr_pct": res["mod_cagr"] * 100,
                "diff_pp": res["cagr_diff_pp"],
            })
    return pd.DataFrame(rows)
