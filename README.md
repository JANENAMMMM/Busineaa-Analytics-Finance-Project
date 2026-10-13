# Finance Project — 다음 거래일 주가 방향 예측

대상일 09:30 이전에 이용 가능한 가격 정보로, 전일 종가 대비 대상일 종가의 변화를 5등급(`0`: 급하락, `1`: 하락, `2`: 보합, `3`: 상승, `4`: 급상승)으로 예측합니다. 모델의 설계와 실험 결과는 함께 제출한 발표자료에 설명되어 있습니다.

## 1. 실행 환경과 파일 준비

Python 3.11 이상과 `uv`를 사용합니다. 아래 명령은 모두 `main.py`가 있는 프로젝트 루트에서 실행합니다.

`uv`가 없다면 먼저 설치합니다.

**Windows PowerShell**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

고정된 의존성을 설치합니다.

```bash
uv sync --locked
```

OneDrive 등에서 하드링크 오류가 발생하면 복사 방식으로 설치합니다.

```bash
uv sync --locked --link-mode copy
```

제공 데이터는 `dataset/`에, 제공된 Alpaca 인증정보 파일은 프로젝트 루트의 `.env`에 배치합니다. `dataset/dataset/`처럼 중첩되지 않도록 합니다.

```text
project_root/
├─ main.py
├─ src/
├─ artifacts/
│  └─ premarket_rule.json
├─ work/b/features/
├─ dataset/
│  ├─ daily.parquet
│  ├─ price.parquet
│  ├─ news.parquet
│  ├─ earnings.parquet
│  ├─ analyst.parquet
│  ├─ symbols.json
│  └─ reddit/
├─ .env
├─ pyproject.toml
└─ uv.lock
```

`.env` 형식은 다음과 같습니다. 수집 명령은 이 파일을 읽으며, 같은 이름의 운영체제 환경변수가 있으면 그 값을 우선 사용합니다.

```dotenv
APCA_API_KEY_ID=제공된_Key_ID
APCA_API_SECRET_KEY=제공된_Secret
```

별도 키가 필요하면 [Alpaca Dashboard](https://app.alpaca.markets/)의 Paper Trading 계정에서 API Key와 Secret을 발급받아 사용합니다.

## 2. 프리마켓 데이터 사전 수집

예측 전에 Alpaca에서 **대상일 09:29까지 종료된 1분봉**을 수집해 종목별 가격으로 축약합니다. 이 단계에는 API 키와 인터넷 연결이 필요합니다. 예측 중에는 외부 API를 호출하지 않습니다.

공개 데이터 전체 기간의 수집 예시입니다. 아래 명령은 PowerShell에서도 한 줄로 실행할 수 있습니다.

```bash
uv run python work/b/features/premarket_rule.py fetch --data dataset --since 2024-08-13 --until 2026-09-12 --as-of 09:29
```

- `--since`와 `--until`은 **기준일** 범위이며, `--until`은 포함하지 않습니다.
- 위 명령은 기준일 2026-09-11까지 포함합니다. 대응하는 대상일 캐시는 2024-08-14~2026-09-14입니다.
- 파일은 `artifacts/premarket_0929/<대상일>.parquet`에 저장됩니다.
- 필요한 날짜·종목의 파일이 준비되어 있으면 수집을 건너뛸 수 있습니다. 재실행 시 기존 파일에 빠진 종목만 추가 요청하므로, 거래가 없는 종목은 다시 조회될 수 있습니다.
- 수집 명령은 정상 처리 시 종료 코드 `0`, API·파일 읽기·저장 실패 시 `1`을 반환합니다. 분봉이 없는 경우와 API 오류는 구분하며, 수집 실패 시 기존 캐시는 보존합니다.

**10월 데이터를 받아 일괄 평가할 때**는 새 데이터를 같은 파일 구성·열 형식으로 준비하고, 위 명령의 `--data`와 기준일 범위를 해당 데이터에 맞게 변경합니다. 월 첫 거래일의 예측에는 이전 달 마지막 거래일이 기준일로 필요할 수 있습니다.

데이터의 거래일 목록에 없는 대상일을 직접 수집하려면 실제 티커를 지정합니다.

```bash
uv run python work/b/features/premarket_rule.py fetch --target-date 2026-10-14 --symbols AMD,MSFT --as-of 09:29
```

`--target-date`는 여러 번 지정할 수 있습니다. 익명 종목 ID는 Alpaca에서 조회할 수 없으므로 실제 티커 대응이 필요합니다. 시험 수집에는 `--store <별도 폴더>`를 사용할 수 있지만, 실제 예측은 기본 경로인 `artifacts/premarket_0929/`를 읽습니다.

Alpaca 명령은 프리마켓 축약 파일만 생성합니다. 교수님 제공 형식의 일봉·시간봉과 필요한 과거 가격 이력은 별도로 준비해야 합니다. 당일 개장 직전의 실시간 수집에는 SIP 데이터 지연·권한 제약이 있을 수 있으며, 위 과거 데이터 일괄 평가와 구분됩니다.

## 3. 모델 확인·예측·평가

모델 로드를 확인합니다. 정상이라면 `Model`이 출력됩니다.

```bash
uv run python -c "from src import load_model; print(type(load_model()).__name__)"
```

출력 형식을 검사합니다. 평가 대상일이 없으면 안내와 함께 종료 코드 `1`을 반환합니다.

```bash
uv run python main.py check --data dataset --split 2026-06-01
```

하루 예측 결과를 저장합니다.

```bash
uv run python main.py predict --data dataset --date 2026-09-10 --out predictions.csv
```

구간을 평가하고 결과를 저장합니다.

```bash
uv run python main.py eval --data dataset --split 2026-06-01 --out evaluation.csv
```

- `--date`는 **기준일**이며, 그 다음 거래일을 예측합니다.
- `--split`은 **예측 대상일** 기준의 평가 시작일입니다. 2026-06-01 평가의 첫 기준일은 2026-05-29입니다.
- 새 데이터 평가 시 `--data`와 날짜를 변경합니다. 데이터에는 기준일·다음 거래일과 피처 계산에 필요한 과거 이력이 있어야 합니다.
- 일부 종목만 실행하려면 `--symbols AMD,MSFT`를 추가합니다. 해당 종목도 입력 데이터에 포함되어야 합니다.
- 하루 예측 출력은 `symbol`, `label` 두 열이며, `label`은 `0~4` 정수입니다. 평가 CSV에는 날짜·정답·예측값도 포함됩니다.

채점 인터페이스는 `load_model()`과 `Model.predict(day)`입니다. 예측 시 재학습, W&B 로그인, LLM 호출은 필요하지 않습니다.

## 4. 제출 모델과 평가 시 처리

최종 모델은 프리마켓 가격 갭을 최근 20거래일 수익률의 표준편차(`vol20`)로 보정하는 규칙입니다.

```text
gap   = (대상일 09:29까지 확정된 가격 / 전일 종가) - 1
gap_z = gap / vol20
score = gap_z
```

제출 파라미터는 `artifacts/premarket_rule.json`에 저장되어 있습니다. `lambda=0`, `a≈0.07607`, `b≈0.25106`이며, 실행 시 JSON의 정밀한 값을 그대로 사용합니다.

| 조건 | label |
|---|---:|
| `score <= -b` | 0 |
| `-b < score <= -a` | 1 |
| `-a < score < a` | 2 |
| `a <= score < b` | 3 |
| `b <= score` | 4 |

프리마켓 파일이나 종목 값이 없거나 사용할 수 없으면 **B0의 시간외 갭 입력값**으로 대체합니다. 분류 파라미터는 제출 모델의 값을 유지하므로, 별도 B0 모델의 최종 라벨과 같다는 뜻은 아닙니다.

평가 시 학습된 파라미터를 변경하지 않습니다. `gap`과 `vol20` 등은 각 예측 시점에 관측 가능한 과거 정보로 계산하며, 대상일 종가·정답이나 평가셋 전체 통계를 이용해 경계를 조정하지 않습니다. 일봉·시간봉은 `known_at` 기준 대상일 09:30 이전 정보만, 프리마켓은 09:29까지 종료된 봉만 사용합니다.

## 5. 홀드아웃 평가 기록

개발 4폴드에서 규칙을 선택한 뒤, 홀드아웃에서는 모델 선택이나 파라미터 조정 없이 독립적으로 평가했습니다. 홀드아웃은 **대상일 2026-06-01~2026-09-14, 73거래일·3,650건**이며, 대응하는 기준일은 **2026-05-29~2026-09-11**입니다.

| 모델 | 홀드아웃 점수 |
|---|---:|
| B0 | 0.5967 |
| 09:15 | 0.5996 |
| 09:28 | 0.6381 |
| **09:29 최종 규칙** | **0.6534** |

위 값은 개발 당시 홀드아웃 평가 기록입니다. 비공개 10월 평가에서는 저장된 파라미터를 그대로 사용하며 추가 학습하지 않습니다.

제출 전 모델 로드·형식 검사·하루 예측·구간 평가를 실행했고, 기존 및 회귀 테스트 68개가 통과했습니다.
