"""
H1: 중도해지(하락장 중 환매) 손실 완화 가설 검증.

100% 주식 포트폴리오 기준으로 하락 구간(episode)을 식별한 뒤, 동일한 구간(날짜 창)에서
주식 비중별 포트폴리오가 실제로 겪는 낙폭(MDD)과 회복 소요일을 비교한다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

from drawdown import identify_episodes


def simulate_exit_losses(
    aligned: pd.DataFrame,
    portfolio_grid: pd.DataFrame,
    weights: list[float],
    threshold: float = -0.10,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    aligned: align_series() 결과 (date 컬럼 기준)
    portfolio_grid: build_portfolio_grid() 결과 (date, w{weight:.2f} 컬럼들)
    반환: (episode x weight 결과 롱포맷 DataFrame, 100%주식 기준 episode 리스트)
    """
    dates = aligned["date"].values
    equity_only = portfolio_grid["w1.00"].to_numpy()
    episodes = identify_episodes(dates, equity_only, threshold=threshold)

    rows = []
    n = len(dates)
    for ei, ep in enumerate(episodes):
        peak_idx = ep["peak_idx"]
        # 탐색 종료 지점: 회복일이 있으면 회복일, 없으면 데이터 끝
        search_end = ep["recovery_idx"] if ep["recovery_idx"] is not None else n - 1

        for w in weights:
            col = f"w{w:.2f}"
            series = portfolio_grid[col].to_numpy()
            peak_val = series[peak_idx]
            window = series[peak_idx:search_end + 1]
            rel = window / peak_val - 1.0
            trough_local = int(np.argmin(rel))
            trough_idx = peak_idx + trough_local
            mdd = rel[trough_local]

            # 회복일: trough 이후 최초로 peak_val 이상 회복하는 시점 탐색(데이터 끝까지)
            recovery_idx = None
            for t in range(trough_idx, n):
                if series[t] >= peak_val:
                    recovery_idx = t
                    break
            recovery_days = (
                (dates[recovery_idx] - dates[trough_idx]) / np.timedelta64(1, "D")
                if recovery_idx is not None else np.nan
            )

            rows.append({
                "episode_id": ei,
                "equity_peak_date": pd.Timestamp(ep["peak_date"]),
                "equity_trough_date": pd.Timestamp(ep["trough_date"]),
                "equity_mdd": ep["mdd"],
                "weight": w,
                "portfolio_peak_date": pd.Timestamp(dates[peak_idx]),
                "portfolio_trough_date": pd.Timestamp(dates[trough_idx]),
                "portfolio_mdd": mdd,
                "recovered": recovery_idx is not None,
                "recovery_days": recovery_days,
            })

    return pd.DataFrame(rows), episodes


def aggregate_by_weight(results: pd.DataFrame) -> pd.DataFrame:
    agg = results.groupby("weight").agg(
        n_episodes=("episode_id", "count"),
        mean_mdd=("portfolio_mdd", "mean"),
        median_mdd=("portfolio_mdd", "median"),
        worst_mdd=("portfolio_mdd", "min"),
        n_recovered=("recovered", "sum"),
        mean_recovery_days=("recovery_days", "mean"),
        median_recovery_days=("recovery_days", "median"),
    ).reset_index()
    return agg


def paired_significance_tests(results: pd.DataFrame, weights: list[float], baseline: float = 1.0) -> pd.DataFrame:
    """각 비중 w에 대해, baseline(기본 100% 주식) 대비 portfolio_mdd 차이의 Wilcoxon signed-rank 검정."""
    pivot_mdd = results.pivot(index="episode_id", columns="weight", values="portfolio_mdd")
    pivot_rec = results.pivot(index="episode_id", columns="weight", values="recovery_days")

    rows = []
    base_mdd = pivot_mdd[baseline]
    base_rec = pivot_rec[baseline]
    for w in weights:
        if w == baseline:
            continue
        diff_mdd = pivot_mdd[w] - base_mdd  # 채권 섞을수록 덜 음수(=완화)면 diff_mdd > 0
        try:
            stat_mdd, p_mdd = stats.wilcoxon(pivot_mdd[w], base_mdd)
        except ValueError:
            stat_mdd, p_mdd = np.nan, np.nan

        rec_pairs = pd.concat([pivot_rec[w], base_rec], axis=1).dropna()
        if len(rec_pairs) >= 3:
            try:
                stat_rec, p_rec = stats.wilcoxon(rec_pairs.iloc[:, 0], rec_pairs.iloc[:, 1])
            except ValueError:
                stat_rec, p_rec = np.nan, np.nan
        else:
            stat_rec, p_rec = np.nan, np.nan

        rows.append({
            "weight": w,
            "mean_mdd_diff_vs_baseline_pp": diff_mdd.mean() * 100,
            "mdd_wilcoxon_p": p_mdd,
            "n_episodes_with_recovery_both": len(rec_pairs),
            "mean_recovery_days_diff_vs_baseline": (rec_pairs.iloc[:, 0] - rec_pairs.iloc[:, 1]).mean() if len(rec_pairs) else np.nan,
            "recovery_wilcoxon_p": p_rec,
        })
    return pd.DataFrame(rows)
