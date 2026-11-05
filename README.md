# Option Pricing Lab

AAPL의 실제 과거 주가로 변동성을 추정하고, 유럽형 콜옵션을 Monte Carlo와 Black–Scholes–Merton으로 평가하는 모의 옵션가치 평가 프로젝트입니다.

## 입력

- Yahoo Finance AAPL, **2024-01-02~2024-12-31**, 가격 252개·로그수익률 251개
- 기준 주가 **250.42 USD**, 조정종가 수익률로 추정한 연 변동성 **22.3238%**
- 가상 조건: 행사가격 250 USD, 만기 0.25년, 연속복리 금리 4%, 배당수익률 0.5%
- 기본 seed 20261105, NumPy PCG64. 원자료 해시와 조회 시각은 `data/source_manifest.json`에 기록합니다.

## 최종 결과

| 방법 | 옵션가치, USD/주 | 표준오차 | 만기 표본 수 |
|---|---:|---:|---|
| Black–Scholes–Merton | 12.419241 | — | 공식 |
| 일반 Monte Carlo | 12.414615 | 0.018262 | 1,000,000 |
| 대칭변량 | 12.399852 | 0.013388 | 1,000,000, 즉 500,000쌍 |
| 제어변량 | 12.418689 | 0.008302 | 본 계산 1,000,000 + 사전 추정 20,000 |

제어변량의 근사 95% 신뢰구간은 **[12.402416, 12.434962]**입니다. 별도 사전 표본으로 계수를 추정하고, 할인한 만기 주가의 알려진 기대값을 이용합니다. 같은 본 계산 표본의 일반 방식 대비 표준오차가 약 **54.6% 감소**했습니다. 독립 수치적분과 공식도 대사했습니다.

금리·배당수익률은 지정 가정이며, 과거 실현변동성을 일정한 미래 변동성 입력으로 사용합니다. MC 신뢰구간은 고정한 모형 아래의 수치적 오차를 나타냅니다.

## 실행

```bash
python3 -m pip install -r requirements.txt
python3 run_experiment.py
python3 -m unittest discover -s tests -v
```

저장된 원자료로 오프라인 재현합니다. 새 자료가 필요할 때만 `python3 download_data.py`를 실행합니다.

- `option_pricing/`: 자료 검증·변동성·BSM·일반 MC·대칭변량·제어변량
- `outputs/results.json`: 입력·환경·전체 결과·독립 수치적분
- `outputs/method_comparison.csv`: 방법별 표본 수와 표준오차
- `outputs/`: 사용 가격, 수렴·seed·민감도 표와 차트
- `tests/`: 알려진 해·경계조건·재현성·제어변량·원자료 검증 **22개 통과**

[Yahoo Finance](https://finance.yahoo.com/quote/AAPL/history/) · [OIC](https://prd-web.optionseducation.org/advancedconcepts/black-scholes-formula) · [NumPy 난수 스트림](https://numpy.org/doc/stable/reference/random/parallel.html)
