# Option Pricing Lab

AAPL의 실제 과거 주가로 변동성을 추정하고, 유럽형 콜옵션을 Monte Carlo와 Black–Scholes–Merton으로 평가하는 모의 옵션가치 평가 프로젝트입니다.

## 입력과 결과

- 자료: Yahoo Finance AAPL, 2024-01-02~2024-12-31, 일별 가격 252개 / 로그수익률 251개
- 기준 주가: 2024-12-31 일반 종가 **250.42 USD**
- 변동성: 조정종가 로그수익률의 표본 표준편차 × √252 = **22.3238%**
- 가상 조건: 행사가격 250 USD, 만기 0.25년, 연속복리 금리 4%, 연속 배당수익률 0.5%
- Black–Scholes–Merton: **12.419241 USD/주**
- Monte Carlo 1,000,000회: **12.414615 USD/주**
- MC 표준오차: 0.018262, 근사 95% 신뢰구간: [12.378822, 12.450409]
- 기본 seed: 20261105, NumPy PCG64

금리와 배당수익률은 해당 기준일의 실제 관측치가 아닌 지정 가정입니다. 조정종가는 2026-11-05 조회본으로, 당시 이용 가능한 데이터 스냅샷을 재현한 백테스트가 아닙니다. 과거 실현변동성을 미래의 일정한 변동성으로 사용하는 가정이며 내재변동성과 구분합니다.

## 실행

Python 3.9 이상에서:

```bash
python3 -m pip install -r requirements.txt
python3 download_data.py  # 최초 다운로드: 네트워크 필요
python3 run_experiment.py  # 이후 저장된 원자료로 오프라인 실행
python3 -m unittest discover -s tests -v
```

실행 확인 환경: Python 3.9.6 / NumPy 1.26.4 / Matplotlib 3.8.3. 원자료 해시와 조회 시각은 `data/source_manifest.json`, 계산 환경과 전체 결과는 `outputs/results.json`에 기록합니다.

## 구조

- `option_pricing/model.py`: 배당 반영 BSM, 정확한 만기 GBM 표본, MC·대칭변량·표준오차
- `option_pricing/data.py`: 원자료 검증, 변동성 추정
- `run_experiment.py`: 표본 수·seed·변동성 비교, CSV와 차트 생성
- `tests/test_model.py`: 알려진 해, 만기·변동성 0, 풋콜패리티, 독립 스칼라 대사, 대칭변량 쌍 단위 표준오차 등 **17개 테스트 통과**
- `blog/16_monte_carlo_option.md`: 벨로그 원고 백업

`outputs/`에는 사용 가격 CSV, 수렴·seed·민감도 표, 주가·시뮬레이션·수렴 차트가 생성됩니다. 원자료는 `data/`에, 생성 결과는 `outputs/`에 정리해 코드와 함께 제공합니다.

## 해석

유럽형 콜은 만기 가격만 필요하므로 일별 경로 대신 위험중립 GBM의 만기 분포를 직접 표본추출합니다. 수렴 비교는 동일한 100만 난수의 앞부분을 사용합니다. MC 신뢰구간은 고정한 모형과 입력 아래의 수치적 추정 오차이며, 미래 주가 범위나 실제 시장가치의 보장 범위가 아닙니다. 대칭변량은 100만 만기 표본을 50만 독립 쌍으로 처리합니다.

## 참고

- [Yahoo Finance AAPL](https://finance.yahoo.com/quote/AAPL/history/)
- [OIC: Black–Scholes Formula](https://prd-web.optionseducation.org/advancedconcepts/black-scholes-formula)
- [Karl Sigman: Geometric Brownian Motion](https://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf)
