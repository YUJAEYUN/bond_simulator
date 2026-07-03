"""낙폭(drawdown) 계산 및 하락 구간(episode) 식별."""
from __future__ import annotations
import numpy as np
import pandas as pd


def rolling_drawdown(values: np.ndarray) -> np.ndarray:
    """각 시점의 낙폭(직전 고점 대비 %, <=0)을 반환."""
    running_max = np.maximum.accumulate(values)
    return values / running_max - 1.0


def max_drawdown(values: np.ndarray) -> float:
    return rolling_drawdown(values).min()


def identify_episodes(dates: np.ndarray, values: np.ndarray, threshold: float = -0.10) -> list[dict]:
    """
    직전 고점 대비 threshold(예: -0.10) 이하로 떨어진 하락 구간들을 식별.
    각 구간: peak_date/idx, trough_date/idx(구간 내 최저점), recovery_date/idx(고점 재돌파, 없으면 None)
    """
    n = len(values)
    running_max = np.maximum.accumulate(values)
    dd = values / running_max - 1.0

    episodes = []
    in_episode = False
    peak_idx = 0
    trough_idx = 0

    i = 1
    while i < n:
        if not in_episode:
            if dd[i] < 0:
                # 새 하락 구간 시작 후보 (직전 고점 인덱스 찾기)
                peak_idx = i - 1
                while peak_idx > 0 and values[peak_idx] < running_max[i]:
                    peak_idx -= 1
                in_episode = True
                trough_idx = i
            i += 1
        else:
            if values[i] < values[trough_idx]:
                trough_idx = i
            if dd[i] >= 0:  # 고점 재돌파 -> 구간 종료
                if dd[trough_idx] <= threshold:
                    episodes.append({
                        "peak_idx": peak_idx, "peak_date": dates[peak_idx], "peak_value": values[peak_idx],
                        "trough_idx": trough_idx, "trough_date": dates[trough_idx], "trough_value": values[trough_idx],
                        "recovery_idx": i, "recovery_date": dates[i],
                        "mdd": dd[trough_idx],
                        "recovery_days": (dates[i] - dates[trough_idx]) / np.timedelta64(1, "D"),
                        "decline_days": (dates[trough_idx] - dates[peak_idx]) / np.timedelta64(1, "D"),
                    })
                in_episode = False
            i += 1

    # 마지막까지 회복 못 한 진행 중 구간
    if in_episode and dd[trough_idx] <= threshold:
        episodes.append({
            "peak_idx": peak_idx, "peak_date": dates[peak_idx], "peak_value": values[peak_idx],
            "trough_idx": trough_idx, "trough_date": dates[trough_idx], "trough_value": values[trough_idx],
            "recovery_idx": None, "recovery_date": None,
            "mdd": dd[trough_idx],
            "recovery_days": None,
            "decline_days": (dates[trough_idx] - dates[peak_idx]) / np.timedelta64(1, "D"),
        })

    return episodes
