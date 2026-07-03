# 채권-주식 혼합 장기투자 가설 검증

S&P500/KOSPI 지수와 미국/한국 10년물 국채 수익률(1996~2026)을 이용해 두 가설을 검증한다.

- **H1 (중도해지 손실 완화)**: 하락장 중 강제로 환매했을 때, 채권을 섞은 포트폴리오가 낙폭(MDD)과
  회복 소요기간을 유의미하게 줄여주는가?
- **H2 (음의 상관관계 / 하락장 방어)**: 주식이 하락하는 날/구간에 채권이 실제로 반대로 움직이는가?

## 폴더 구조

```
simulation/
├── data/                  # 정리된 원본 시계열 (date, close, change_pct)
│   ├── sp500.csv
│   ├── kospi.csv
│   ├── us10y_yield.csv    # 미국 10년물 국채 만기수익률(%)
│   └── kr10y_yield.csv    # 한국 10년물 국채 만기수익률(%)
├── src/
│   ├── data_loader.py         # investing.com 포맷 CSV 파서
│   ├── build_dataset.py       # 원본 업로드 파일 -> data/*.csv 생성 스크립트
│   ├── equity_tr.py           # 주가지수 누적수익지수(TR) 생성
│   ├── bond_tr.py              # 채권 수익률(YTM) -> 총수익지수 재구성
│   ├── portfolio.py           # 비중별/리밸런싱 주기별 혼합 포트폴리오 TR
│   ├── drawdown.py            # 낙폭·하락구간(episode) 식별
│   ├── exit_loss_simulation.py # H1: 중도해지 손실 시뮬레이션 + 유의성 검정
│   ├── correlation_analysis.py # H2: 전체기간/위기구간/롤링 상관관계
│   ├── best_days_removal.py   # 기존 Best-Days 로직(Python 이식) + 비중별 확장
│   └── run_all.py             # 전체 파이프라인 실행 (진입점)
├── output/                # run_all.py 실행 결과 (CSV/PNG/summary_report.md)
└── web/                   # 기존 "Best Days 제거 시뮬레이터" (순수 JS, 브라우저용)
```

## 실행 방법

```bash
pip install -r requirements.txt
cd src
python3 build_dataset.py   # (최초 1회) 업로드 원본 -> data/*.csv 정리
python3 run_all.py         # 전체 분석 실행, output/ 에 결과 생성
```

`web/` 폴더는 기존 KOSPI/S&P500 "최고의 날 제거" 브라우저 시뮬레이터로, 독립적으로
`python3 -m http.server` 로 띄워서 사용할 수 있다 (인터넷 연결 필요 — Chart.js/PapaParse를
CDN에서 로드).

## 핵심 방법론

- **채권 총수익지수 재구성**: 보유 데이터가 ETF 가격+분배금이 아닌 constant-maturity
  국채 만기수익률(YTM)이므로, 매일 "잔존만기 10년 액면채권(표면금리=전일 수익률)"을
  당일 수익률로 재할인하는 방식(par bond repricing)으로 가격수익 + 이자캐리를 반영한
  총수익지수를 만든다. 상세 가정/한계는 `src/bond_tr.py` 상단 주석 참고.
- **포트폴리오**: 주식 비중 100/80/60/40/20/0%, 리밸런싱 없음(buy&hold) vs 연 1회 비교.
- **하락구간(episode)**: 100% 주식 지수의 직전 고점 대비 -10% 이하 낙폭 구간을 식별,
  동일 날짜 창에서 비중별 포트폴리오의 실제 손실/회복기간을 비교.
- **통계 검정**: Wilcoxon signed-rank test (paired, 하락구간 단위).

## 산출물 (output/)

| 파일 | 내용 |
|---|---|
| `tr_index_reconstructed.csv` | 채권 TR 재구성 결과 (수익률/일별수익률/TR지수) |
| `drawdown_by_weight.csv` / `.png` | 비중별 평균/최악 MDD, 회복기간 |
| `exit_loss_simulation.csv` | 하락구간 x 비중 단위 손실/회복 원자료 |
| `exit_loss_episodes_{pair}.png` | 대표 하락구간(최악 3개)의 비중별 손익 곡선 |
| `significance_tests.csv` | 100% 주식 대비 비중별 Wilcoxon 검정 결과 |
| `correlation_analysis.csv` | 전체기간 vs 위기구간(하위5%) 상관계수 |
| `correlation_yearly.csv` | 연도별 상관계수 |
| `correlation_{pair}.png` | 60일 롤링 상관계수 + 일별수익률 산점도 |
| `best_days_by_weight.csv` | Best-days 제거 로직을 비중별 포트폴리오로 확장한 결과 |
| `summary_report.md` | 위 결과를 종합한 결론 (통계적 유의성 포함) |

## 핵심 결론 요약

- **H1은 두 페어(S&P500/미국채, KOSPI/한국채) 모두 강하게 채택**된다. 채권 비중이
  높아질수록 낙폭과 회복기간이 모든 비중 구간에서 통계적으로 유의하게(p<0.01) 감소한다.
- **H2는 페어별로 갈린다.** 미국 페어는 전체기간·위기구간 모두 유의한 음의 상관관계를
  보이지만, 한국 페어는 전체기간 상관계수가 -0.04로 사실상 0에 가깝고 위기구간에서는
  통계적으로 유의하지 않다(p=0.46).
- 자세한 수치와 해석은 `output/summary_report.md` 참고.
