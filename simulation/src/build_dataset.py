"""원본 업로드 CSV를 이어붙여 simulation/data/ 아래 표준 포맷(date, close, change_pct)으로 저장."""
from __future__ import annotations
import os
from data_loader import load_series

UPLOAD_DIR = "/root/.claude/uploads/16063ece-7ed2-5c79-a1ff-acccf58cb1d8"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SOURCES = {
    "kospi.csv": [
        "e5b842c1-kospi_1996_2015.csv",
        "007e180a-kospi_2015_2026.csv",
    ],
    "sp500.csv": [
        "8f67dcd4-SP_500_______1996010220151109.csv",
        "9abf4d82-SP_500________2015111020260527.csv",
    ],
    "kr10y_yield.csv": [
        "b198acaa-___10______________2000102620200515.csv",
        "8a8dcb78-___10______________2020051620260527.csv",
    ],
    "us10y_yield.csv": [
        "218ffa66-___10______________________1996010220150602.csv",
        "7debfa1d-___10______________________2015060320260527.csv",
    ],
}


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for out_name, files in SOURCES.items():
        paths = [os.path.join(UPLOAD_DIR, f) for f in files]
        df = load_series(paths)
        out_path = os.path.join(DATA_DIR, out_name)
        df.to_csv(out_path, index=False)
        print(f"{out_name}: {len(df)} rows, {df['date'].min().date()} ~ {df['date'].max().date()} -> {out_path}")


if __name__ == "__main__":
    main()
