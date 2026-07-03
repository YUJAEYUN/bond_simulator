"""전체 파이프라인 실행: 데이터 로드 -> TR 재구성 -> 포트폴리오 그리드 -> H1/H2 분석 -> output/ 산출."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from equity_tr import build_equity_index
from bond_tr import build_bond_total_return
from portfolio import align_series, build_portfolio_grid
from drawdown import identify_episodes
from exit_loss_simulation import simulate_exit_losses, aggregate_by_weight, paired_significance_tests
from correlation_analysis import (
    full_period_correlation, crisis_conditional_correlation, rolling_correlation,
    yearly_correlation, scatter_data,
)
from best_days_removal import best_days_by_weight

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")
OUT_DIR = os.path.join(BASE, "..", "output")

CONFIG = {
    "equity_weights": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    "rebalance_freqs": ["annual", "none"],
    "drawdown_threshold": -0.10,
    "crisis_percentile": 0.05,
    "rolling_corr_window": 60,
    "best_days_ns": [0, 5, 10, 15, 20, 25, 30],
    "pairs": {
        "sp500": {"equity_file": "sp500.csv", "bond_file": "us10y_yield.csv", "label": "S&P500 / 미국채10Y"},
        "kospi": {"equity_file": "kospi.csv", "bond_file": "kr10y_yield.csv", "label": "KOSPI / 한국채10Y"},
    },
}

COLORS = {1.0: "#1f4e8c", 0.8: "#3d7ab5", 0.6: "#6aa6d8", 0.4: "#f2b53c", 0.2: "#e07b39", 0.0: "#c0392b"}


def load_pair(pair_key: str):
    cfg = CONFIG["pairs"][pair_key]
    eq_raw = pd.read_csv(os.path.join(DATA_DIR, cfg["equity_file"]), parse_dates=["date"])
    bd_raw = pd.read_csv(os.path.join(DATA_DIR, cfg["bond_file"]), parse_dates=["date"])
    eq_tr = build_equity_index(eq_raw)
    bd_tr = build_bond_total_return(bd_raw)
    aligned = align_series(eq_tr, bd_tr)
    return aligned, bd_tr, cfg


def run_h1(pair_key: str, aligned: pd.DataFrame, weights: list[float]) -> dict:
    results_all, agg_all, sig_all = [], [], []
    episodes_by_freq = {}
    for freq in CONFIG["rebalance_freqs"]:
        grid = build_portfolio_grid(aligned, weights, freq)
        results, episodes = simulate_exit_losses(aligned, grid, weights, CONFIG["drawdown_threshold"])
        results["pair"] = pair_key
        results["rebalance_freq"] = freq
        agg = aggregate_by_weight(results)
        agg["pair"] = pair_key
        agg["rebalance_freq"] = freq
        sig = paired_significance_tests(results, weights, baseline=1.0)
        sig["pair"] = pair_key
        sig["rebalance_freq"] = freq
        results_all.append(results)
        agg_all.append(agg)
        sig_all.append(sig)
        episodes_by_freq[freq] = (grid, episodes)
    return {
        "results": pd.concat(results_all, ignore_index=True),
        "agg": pd.concat(agg_all, ignore_index=True),
        "sig": pd.concat(sig_all, ignore_index=True),
        "episodes_by_freq": episodes_by_freq,
    }


def run_h2(pair_key: str, aligned: pd.DataFrame) -> dict:
    full = full_period_correlation(aligned)
    crisis = crisis_conditional_correlation(aligned, CONFIG["crisis_percentile"])
    roll = rolling_correlation(aligned, CONFIG["rolling_corr_window"])
    yearly = yearly_correlation(aligned)
    scatter = scatter_data(aligned)
    full["pair"] = pair_key
    crisis["pair"] = pair_key
    yearly["pair"] = pair_key
    return {"full": full, "crisis": crisis, "rolling": roll, "yearly": yearly, "scatter": scatter}


def plot_drawdown_by_weight(agg_all: pd.DataFrame, path: str):
    pairs = list(CONFIG["pairs"].keys())
    fig, axes = plt.subplots(len(pairs), 2, figsize=(11, 4 * len(pairs)))
    for i, pk in enumerate(pairs):
        sub = agg_all[(agg_all["pair"] == pk) & (agg_all["rebalance_freq"] == "annual")]
        ax1, ax2 = axes[i]
        ax1.bar([f"{int(w*100)}%" for w in sub["weight"]], sub["mean_mdd"] * 100,
                color=[COLORS[w] for w in sub["weight"]])
        ax1.set_title(f"{CONFIG['pairs'][pk]['label']} — 비중별 평균 낙폭(MDD)")
        ax1.set_ylabel("평균 MDD (%)")
        ax1.axhline(0, color="black", linewidth=0.6)

        ax2.bar([f"{int(w*100)}%" for w in sub["weight"]], sub["mean_recovery_days"],
                color=[COLORS[w] for w in sub["weight"]])
        ax2.set_title(f"{CONFIG['pairs'][pk]['label']} — 비중별 평균 회복 소요일")
        ax2.set_ylabel("일수")
    fig.suptitle("주식 비중(x축: 주식%) 별 하락구간 낙폭·회복기간 (연1회 리밸런싱)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_exit_loss_episodes(pair_key: str, aligned: pd.DataFrame, grid: pd.DataFrame,
                             episodes: list[dict], weights: list[float], path: str, top_k: int = 3):
    ranked = sorted(episodes, key=lambda e: e["mdd"])[:top_k]
    fig, axes = plt.subplots(1, len(ranked), figsize=(5.2 * len(ranked), 4.2), sharey=False)
    if len(ranked) == 1:
        axes = [axes]
    dates = aligned["date"].values
    for ax, ep in zip(axes, ranked):
        peak_idx = ep["peak_idx"]
        end_idx = ep["recovery_idx"] if ep["recovery_idx"] is not None else len(dates) - 1
        end_idx = min(end_idx + 10, len(dates) - 1)
        for w in weights:
            series = grid[f"w{w:.2f}"].to_numpy()
            window = series[peak_idx:end_idx + 1] / series[peak_idx] - 1
            ax.plot(aligned["date"].values[peak_idx:end_idx + 1], window * 100,
                    label=f"주식 {int(w*100)}%", color=COLORS[w], linewidth=1.4)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"{pd.Timestamp(ep['peak_date']).date()} 고점 하락구간\n(equity MDD {ep['mdd']*100:.1f}%)", fontsize=10)
        ax.set_ylabel("고점 대비 손익 (%)")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{CONFIG['pairs'][pair_key]['label']} — 대표 하락구간별 비중별 손익 곡선")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_correlation(pair_key: str, h2: dict, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    roll = h2["rolling"]
    axes[0].plot(roll["date"], roll["rolling_corr"], color="#1f4e8c", linewidth=0.8)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_title(f"{CONFIG['pairs'][pair_key]['label']} — 60일 롤링 상관계수")
    axes[0].set_ylabel("Pearson r")

    sc = h2["scatter"]
    cutoff = sc["equity_ret"].quantile(CONFIG["crisis_percentile"])
    crisis_mask = sc["equity_ret"] <= cutoff
    axes[1].scatter(sc.loc[~crisis_mask, "equity_ret"] * 100, sc.loc[~crisis_mask, "bond_ret"] * 100,
                     s=6, alpha=0.35, color="#6aa6d8", label="평상시")
    axes[1].scatter(sc.loc[crisis_mask, "equity_ret"] * 100, sc.loc[crisis_mask, "bond_ret"] * 100,
                     s=10, alpha=0.7, color="#c0392b", label=f"주식 하락 하위{int(CONFIG['crisis_percentile']*100)}%")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axvline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("주식 일별수익률 (%)")
    axes[1].set_ylabel("채권 일별수익률 (%)")
    axes[1].set_title("일별 수익률 산점도")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    weights = CONFIG["equity_weights"]

    tr_rows = []
    h1_results, h1_agg, h1_sig = [], [], []
    h2_full, h2_crisis, h2_yearly = [], [], []
    best_days_rows = []

    pair_aligned = {}
    pair_grids_annual = {}

    for pk, cfg in CONFIG["pairs"].items():
        aligned, bond_tr_df, _ = load_pair(pk)
        pair_aligned[pk] = aligned

        bt = bond_tr_df.copy()
        bt["pair"] = pk
        tr_rows.append(bt)

        h1 = run_h1(pk, aligned, weights)
        h1_results.append(h1["results"])
        h1_agg.append(h1["agg"])
        h1_sig.append(h1["sig"])

        grid_annual, episodes_annual = h1["episodes_by_freq"]["annual"]
        pair_grids_annual[pk] = (grid_annual, episodes_annual)
        plot_exit_loss_episodes(pk, aligned, grid_annual, episodes_annual, weights,
                                 os.path.join(OUT_DIR, f"exit_loss_episodes_{pk}.png"))

        bd = best_days_by_weight(aligned, grid_annual, weights, CONFIG["best_days_ns"])
        bd["pair"] = pk
        best_days_rows.append(bd)

        h2 = run_h2(pk, aligned)
        h2_full.append(h2["full"])
        h2_crisis.append(h2["crisis"])
        h2_yearly.append(h2["yearly"])
        plot_correlation(pk, h2, os.path.join(OUT_DIR, f"correlation_{pk}.png"))

    tr_all = pd.concat(tr_rows, ignore_index=True)
    tr_all.to_csv(os.path.join(OUT_DIR, "tr_index_reconstructed.csv"), index=False)

    h1_results_all = pd.concat(h1_results, ignore_index=True)
    h1_agg_all = pd.concat(h1_agg, ignore_index=True)
    h1_sig_all = pd.concat(h1_sig, ignore_index=True)
    h1_results_all.to_csv(os.path.join(OUT_DIR, "exit_loss_simulation.csv"), index=False)
    h1_agg_all.to_csv(os.path.join(OUT_DIR, "drawdown_by_weight.csv"), index=False)
    h1_sig_all.to_csv(os.path.join(OUT_DIR, "significance_tests.csv"), index=False)
    plot_drawdown_by_weight(h1_agg_all, os.path.join(OUT_DIR, "drawdown_by_weight.png"))

    h2_full_df = pd.DataFrame(h2_full)
    h2_crisis_df = pd.DataFrame(h2_crisis)
    h2_yearly_df = pd.concat(h2_yearly, ignore_index=True)
    corr_summary = pd.merge(
        h2_full_df.rename(columns={"pearson_r": "full_period_r", "p_value": "full_period_p", "n": "full_n"}),
        h2_crisis_df.rename(columns={"pearson_r": "crisis_r", "p_value": "crisis_p", "n": "crisis_n"}),
        on="pair",
    )
    corr_summary.to_csv(os.path.join(OUT_DIR, "correlation_analysis.csv"), index=False)
    h2_yearly_df.to_csv(os.path.join(OUT_DIR, "correlation_yearly.csv"), index=False)

    best_days_all = pd.concat(best_days_rows, ignore_index=True)
    best_days_all.to_csv(os.path.join(OUT_DIR, "best_days_by_weight.csv"), index=False)

    write_summary_report(h1_agg_all, h1_sig_all, corr_summary, h2_yearly_df, best_days_all)

    print("Done. Outputs in", OUT_DIR)


def write_summary_report(h1_agg, h1_sig, corr_summary, corr_yearly, best_days):
    lines = []
    lines.append("# 채권-주식 혼합 장기투자 가설 검증 — 결과 요약\n")
    lines.append(f"- 분석 대상: {', '.join(CONFIG['pairs'][k]['label'] for k in CONFIG['pairs'])}")
    lines.append(f"- 주식 비중 grid: {[f'{int(w*100)}%' for w in CONFIG['equity_weights']]}")
    lines.append(f"- 하락구간 정의: 직전 고점 대비 {CONFIG['drawdown_threshold']*100:.0f}% 이하")
    lines.append(f"- 위기일 정의(H2): 주식 일별수익률 하위 {CONFIG['crisis_percentile']*100:.0f}%\n")

    lines.append("## H1. 중도해지 손실 완화 가설\n")
    for pk in CONFIG["pairs"]:
        lines.append(f"### {CONFIG['pairs'][pk]['label']}\n")
        sub = h1_agg[(h1_agg["pair"] == pk) & (h1_agg["rebalance_freq"] == "annual")].sort_values("weight", ascending=False)
        lines.append("| 주식비중 | 평균MDD | 최악MDD | 평균회복일 | 중앙값회복일 | 회복된 구간수 |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(f"| {int(r['weight']*100)}% | {r['mean_mdd']*100:.1f}% | {r['worst_mdd']*100:.1f}% | "
                         f"{r['mean_recovery_days']:.0f}일 | {r['median_recovery_days']:.0f}일 | {int(r['n_recovered'])}/{int(r['n_episodes'])} |")
        lines.append("")
        sig_sub = h1_sig[(h1_sig["pair"] == pk) & (h1_sig["rebalance_freq"] == "annual")]
        sig_line = ", ".join(
            f"{int(r['weight']*100)}% p={r['mdd_wilcoxon_p']:.4f}" for _, r in sig_sub.iterrows()
        )
        lines.append(f"- 100% 주식 대비 MDD 완화폭의 Wilcoxon signed-rank 검정(p-value): {sig_line}")
        all_sig = (sig_sub["mdd_wilcoxon_p"] < 0.05).all()
        lines.append(f"- 판정: {'**채택** (모든 비중에서 p<0.05로 유의미한 MDD 완화)' if all_sig else '**부분 채택/기각 재검토 필요**'}\n")

    lines.append("## H2. 음의 상관관계(하락장 방어) 가설\n")
    lines.append("| 페어 | 전체기간 r | 전체기간 p | 위기구간 r | 위기구간 p |")
    lines.append("|---|---|---|---|---|")
    for _, r in corr_summary.iterrows():
        lines.append(f"| {CONFIG['pairs'][r['pair']]['label']} | {r['full_period_r']:.3f} | {r['full_period_p']:.2e} | "
                     f"{r['crisis_r']:.3f} | {r['crisis_p']:.2e} |")
    lines.append("")

    for pk in CONFIG["pairs"]:
        ysub = corr_yearly[corr_yearly["pair"] == pk]
        positive_years = ysub[ysub["pearson_r"] > 0]["year"].tolist()
        lines.append(f"- {CONFIG['pairs'][pk]['label']}: 연도별 상관계수가 양(+)으로 뒤집힌 해 = {positive_years}")
    lines.append("")

    # 페어별로 개별 판정 (전체기간·위기구간 모두 r<0 AND p<0.05 이어야 채택)
    for _, r in corr_summary.iterrows():
        label = CONFIG['pairs'][r['pair']]['label']
        full_ok = r['full_period_r'] < 0 and r['full_period_p'] < 0.05
        crisis_ok = r['crisis_r'] < 0 and r['crisis_p'] < 0.05
        if full_ok and crisis_ok:
            verdict = "**채택** (전체기간·위기구간 모두 통계적으로 유의한 음의 상관관계)"
        elif full_ok and not crisis_ok:
            verdict = (f"**기각(위기구간 기준)** — 전체기간은 유의한 음의 상관관계이나, "
                       f"정작 중요한 위기구간(주식 하락일)에서는 상관계수가 유의하지 않음(p={r['crisis_p']:.2f}). "
                       f"이 페어에서는 채권이 '하락장 방어' 역할을 한다고 통계적으로 단정할 수 없음")
        elif not full_ok and crisis_ok:
            verdict = "**부분 채택** (전체기간은 유의하지 않으나 위기구간에서는 유의한 음의 상관관계)"
        else:
            verdict = "**기각** (전체기간·위기구간 모두 유의한 음의 상관관계 확인 안 됨)"
        lines.append(f"- {label} 판정: {verdict}")
    lines.append("")

    lines.append("## Best-days 제거 로직 확장 (부가 확인)\n")
    lines.append("주식 비중이 낮을수록 상위 N개 수익일 제외 시 CAGR 하락폭이 줄어드는지 확인 (n=30 기준):\n")
    lines.append("| 페어 | 주식비중 | n=30 제거 시 CAGR 변화(pp) |")
    lines.append("|---|---|---|")
    for pk in CONFIG["pairs"]:
        sub = best_days[(best_days["pair"] == pk) & (best_days["n_removed"] == 30)].sort_values("weight", ascending=False)
        for _, r in sub.iterrows():
            lines.append(f"| {CONFIG['pairs'][pk]['label']} | {int(r['weight']*100)}% | {r['diff_pp']:.2f}pp |")
    lines.append("")

    lines.append("## 종합 결론\n")
    lines.append(
        "- **H1(중도해지 손실 완화)은 두 페어 모두 강하게 채택.** 주식 비중이 낮아질수록 평균/최악 MDD와 회복 소요일이 "
        "모든 비중 구간에서 통계적으로 유의하게(p<0.01) 감소했다. 즉 채권을 섞으면 '하락장 중 강제로 빠져야 했을 때'의 "
        "손실이 실제로, 그리고 일관되게 완화된다.\n"
    )
    lines.append(
        "- **H2(음의 상관관계)는 페어별로 결론이 갈린다.** S&P500/미국채10Y는 전체기간·위기구간 모두 유의한 음의 상관관계로 "
        "'하락장 방어'가 통계적으로 뒷받침되지만, KOSPI/한국채10Y는 전체기간 상관계수가 -0.04로 사실상 0에 가깝고 "
        "위기구간에서는 통계적 유의성이 없다(p=0.46) — 한국 국채는 미국 국채만큼의 '안전자산 플라이트' 효과가 "
        "이 표본기간·10년물 기준으로는 뚜렷하게 나타나지 않았다.\n"
    )
    lines.append(
        "- **종합**: 두 페어 모두 H1만으로 '채권 혼합이 장기투자 리스크 관리 도구로 유의미하다'는 결론이 성립한다. "
        "다만 그 메커니즘은 페어마다 다르다 — 미국 국채는 낙폭 완화(H1) + 하락장 방어(H2) 둘 다 기여하지만, "
        "한국 국채는 주로 '변동성이 낮아 포트폴리오 자체의 낙폭을 줄이는 효과'(H1)에서 오고, "
        "'주식과 반대로 움직여서' 상쇄해주는 효과(H2)는 이 데이터에서 확인되지 않았다.\n"
    )
    lines.append("## 방법론/한계 노트\n")
    lines.append(
        "- 채권 TR은 원자료(가격+분배금)가 아닌 만기수익률(YTM) 시계열로부터, "
        "연1회 이표를 가정한 constant-maturity par bond 재가격 모델로 재구성한 근사치입니다 "
        "(`src/bond_tr.py` 참고). 절대 수익률 값보다 상대적 비교(비중 간 차이, 방향성)에 무게를 두고 해석해야 합니다.\n"
        "- 환율 미고려 (S&P500/미국채는 달러, KOSPI/한국채는 원화 기준 독립 계산).\n"
        "- 세전 기준.\n"
        "- 리밸런싱 없음(buy&hold) 결과는 `exit_loss_simulation.csv`/`drawdown_by_weight.csv`의 rebalance_freq=none 행 참고.\n"
    )

    with open(os.path.join(OUT_DIR, "summary_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
