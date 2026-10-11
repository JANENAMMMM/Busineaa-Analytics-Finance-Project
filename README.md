# Business Analytics Finance Project — 팀 A·B·C (남재은·김민영·안채민)

다음 거래일 종가 등락을 5단계(0 급하락 ~ 4 급상승)로 예측합니다.
제출 모델은 **프리마켓 09:28 가격으로 만든 갭 규칙**입니다. 학습된 상수 6개만 쓰는 고정 규칙이라 평가 때 학습하지 않습니다.

## 1. 한눈에 보는 실행 순서

```bash
uv sync                                                    # ① 환경 (uv.lock 그대로)
# ② 평가 기간 프리마켓 파일 받기 — 아래 3절. 키 필요, 한 번만
uv run python work/b/features/premarket_rule.py fetch --since <첫 기준일> --until <마지막 기준일 다음 날> --as-of 09:28
uv run python main.py check                                # ③ 형식 확인 (선택)
uv run python main.py predict --date <기준일>              # ④ 예측 (채점 코드가 하는 일)
```

②를 건너뛰어도 예측은 오류 없이 돌아갑니다. 다만 그 날짜는 아래 2절의 대체 규칙에 따라 **기존 B0 갭 규칙**으로 예측되어 점수가 낮아집니다(개발 구간 기준 약 −0.04).

## 2. 모델이 하는 일

1. 교수님 데이터로 B0 갭을 만듭니다: `gap_z = (마지막 시간외 가격 ÷ 기준일 종가 − 1) ÷ vol20`, `vol20`은 그 종목의 최근 20거래일 일간 수익률 표준편차입니다.
2. `artifacts/premarket_0928/<대상일>.parquet`가 있으면, 갭 가격을 **대상일 09:28까지 끝난 마지막 1분봉 종가**(`px_asof`)로 바꿉니다.
   - 단위 확인: 그 파일의 08:30 가격(`px_0830`)이 교수님 데이터의 08:00 봉 가격과 2% 넘게 다르면 그 종목은 바꾸지 않습니다.
   - 파일·종목이 없어도 그 종목만 B0 갭을 그대로 씁니다(오류 없음).
3. 갭 세기 `s = gap_z × exp(−λ·z(ext_range_z))`를 고정 경계 a, b로 자릅니다: |s| < a 보합, a ≤ |s| < b 상승/하락, |s| ≥ b 급상승/급하락.

09:28 가격은 예측 시점(09:30) 전에 이미 확정된 정보입니다. 시간봉의 `known_at` 규칙과 같게, **봉이 끝난 시각 ≤ 09:28**인 봉만 씁니다. 저장된 23,632행 전부 이 조건을 만족하는지 검사했습니다(`reports/premarket_submission_check.json`).

## 3. 실행 전에 할 일: 프리마켓 파일 받기

평가 기간의 각 대상일마다 파일이 하나씩 필요합니다. 무료 Alpaca 시장 데이터 계정의 과거 1분봉으로 받으며 결제는 필요 없습니다.

**① 키 설정** (https://alpaca.markets 무료 가입 → API Keys)

```powershell
# Windows PowerShell (설정 후 터미널 재시작)
setx APCA_API_KEY_ID "발급받은_Key_ID"
setx APCA_API_SECRET_KEY "발급받은_Secret"
```
```bash
# macOS / Linux
export APCA_API_KEY_ID="발급받은_Key_ID"
export APCA_API_SECRET_KEY="발급받은_Secret"
```

**② 받기** — `--since`·`--until`은 **기준일**이고 `--until`은 그 날을 **포함하지 않습니다**. 평가 데이터의 첫 기준일과 마지막 기준일 다음 날을 넣으면, 각 기준일의 다음 거래일(대상일) 파일이 생깁니다.

```bash
uv run python work/b/features/premarket_rule.py fetch --since 2026-10-01 --until 2026-11-01 --as-of 09:28
# 기준일 목록은 main.py 옆의 dataset/ 에서 읽습니다
```

- 종목은 평가 데이터의 `day.symbols`에서 읽습니다. 분봉을 붙이려면 **실제 티커**여야 합니다(익명 ID면 그 종목은 B0로 대체).
- 이미 받은 날짜는 건너뜁니다. 원본 1분봉은 저장하지 않고 종목당 한 줄만 남깁니다.
- 결과: `artifacts/premarket_0928/YYYY-MM-DD.parquet` (파일 이름 = **대상일**)

| 열 | 뜻 |
|---|---|
| `symbol` | 티커 |
| `px_0830` | 08:30까지 끝난 마지막 봉 종가 (단위 확인용) |
| `px_asof` | 09:28까지 끝난 마지막 봉 종가 (갭 가격) |
| `t_asof` | 그 봉의 시작 시각 (ET) |
| `vol_pre` | 08:00~09:28 거래량 (기록용, 규칙에는 안 씀) |

공개 기간(대상일 2024-08-14 ~ 2026-09-14) 파일 522개는 ZIP에 들어 있습니다.

## 4. 학습셋에서 정한 값 vs 평가 때 행마다 하는 처리

**학습셋에서 계산해 고정한 값** — `artifacts/premarket_rule.json`. 평가 때 다시 계산하지 않습니다.

| 값 | 뜻 | 값 |
|---|---|---|
| `med`, `sd` | `ext_range_z`를 표준화하는 중앙값·척도 | 1.4675, 4.9708 |
| `lam` | 시간외 가격 범위가 클 때 갭을 덜 믿는 정도 | 0.3 |
| `a`, `b` | 보합/상승·하락/급등락 경계 | 0.0726, 0.3092 |
| `as_of` | 몇 시 가격을 쓸지 | 09:28 |

공개된 정답 전체(대상일 2024-08-14 ~ 2026-09-14, 522일, 50종목)로 맞췄습니다. 근거와 입력 해시: `artifacts/premarket_rule.provenance.json`.

**평가 때 행마다 그대로 적용하는 처리** — 다른 평가 행이나 정답을 보지 않습니다.
- 그 종목의 기준일 종가, 최근 20거래일 수익률로 `vol20` 계산 (그 종목 자신의 과거만 씀)
- 그 종목·그 대상일의 프리마켓 한 줄 읽기, 2% 단위 확인
- 위 고정 상수로 표준화하고 고정 경계로 자르기

평균·표준편차·분위수·경계 등 여러 행을 묶어 정하는 값은 평가 데이터로 계산하지 않습니다.

## 5. 환경

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/). 실행에 필요한 패키지는 전부 `pyproject.toml`에 있고 `uv.lock`으로 고정돼 있습니다.
- 평가에 쓰는 코드: `main.py`, `src/`, `artifacts/`, `work/b/features/premarket_rule.py`(받기 전용).
- `dataset/`은 ZIP에 없습니다. 교수님 데이터를 `main.py` 옆에 두거나 `--data`로 지정하세요.

## 6. 검증 기록 (참고)

| 모델 | 개발 4폴드 평균 | 홀드아웃 73일 (한 번만 사용) |
|---|---:|---:|
| B0 (교수님 데이터 갭) | 0.4469 | 0.5967 |
| 프리마켓 09:15 | 0.4635 | 0.5996 |
| **프리마켓 09:28 (제출)** | **0.4875** (민영 측정) | **0.6381** (B0 대비 95% 구간 +0.018 ~ +0.068) |

- 시각(09:28)은 홀드아웃을 보기 전에 정했습니다. 홀드아웃 숫자는 5/11까지로 학습한 규칙을 6/1~9/14에 한 번 적용한 값입니다. 그 뒤 공개 데이터 전체로 다시 맞춰 제출했습니다.
- 홀드아웃이 개발보다 높은 것은 그 기간에 갭이 방향을 잘 맞힌 구간 효과입니다. B0도 같이 올랐습니다.
- 실험 기록: `docs/worklog/`, `reports/`, W&B `BI_finance_project/finance-direction`.

## 7. 재현 (채점에는 필요 없음)

```bash
uv run python scripts/train_premarket_submission.py   # 고정 상수 다시 만들기 (공개 정답 사용, 평가 때 실행 금지)
uv run python scripts/premarket_holdout_check.py      # 홀드아웃 1회 검사
uv run python scripts/premarket_submission_check.py   # src/model.py == 규칙, 파일 없을 때 == B0, look-ahead 0
```
