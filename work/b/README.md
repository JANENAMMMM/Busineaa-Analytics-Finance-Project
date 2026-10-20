# work/b — B 파트 (정형 데이터: daily · price · earnings · analyst)

담당 민영. 평가는 재은님 폴드(`lab/folds.json`) 기준.

```
work/b/
├─ features/   ← 팀원이 볼 것: 고른 피처, 전달용 코드, 확인 스크립트
├─ eda/        B 데이터 EDA: 노트북 3개, 그림, 보고서
├─ exp/        피처 실험 r01 ~ r12 (1~12차) 와 결과 기록
└─ early/      초기 작업 (A 폴드 이전, B 홀드아웃 기준 베이스라인·V1 점검)
```

## 팀원이 볼 것 — `features/`

| 파일 | 내용 |
|---|---|
| **[`features/minyoung_features.md`](features/minyoung_features.md)** | 고른 피처, 추출 방식, 결측, 선택 이유, 시도한 피처 102개, 팀 피처 규칙 확인 |
| **[`features/gapz_rule.py`](features/gapz_rule.py)** | 전달용 코드. 피처 계산 · 학습 · 예측 한 파일 (numpy, pandas 만) |
| [`features/check_gapz_rule.py`](features/check_gapz_rule.py) | `gapz_rule.py` 확인 (피처 값, 누수, A 폴드 점수 재현) |
| [`features/check_rules.py`](features/check_rules.py) | 팀 피처 규칙 1~8 확인 (누수 검사, 모양, 종목 ID, train 에서만 fit, `src/model.py` 붙여 넣고 `main.py check`) |
| [`features/features_b.py`](features/features_b.py) | B 피처 61개 전체 `build_b(day)`, 추가 후보 11개 `build_b_extra(day)` |

## B 결론

- **고른 피처: `gap_z`, `ext_range_z` 2개**
  - `gap_z` = 시간외 갭 ÷ 20일 변동성 (방향 정보의 거의 전부)
  - `ext_range_z` = 시간외 고저 폭 ÷ 20일 변동성 (급등락 칸까지 갈지를 조금 조정, 효과 작음 — "출렁인 갭은 잡음"이라는 처음 설명은 10/7 정정, `docs/review/03_규칙_설명가능성.md`)
- 결측률 둘 다 0.22% (2026-01-30 하루, 원본 시간외 봉 누락)
- **기준 점수: gap_z 규칙 + ext_range_z = 4폴드 평균 0.447** (fold4 0.515, 처음 보는 종목 0.363)
  - 재은님 기준선(전일 종가 기준 시간외 갭 규칙, `eda/cv_check_ovn_gap.py`) 0.433 과 같은 경계 후보·같은 폴드로 비교
  - 베이스라인 모델이 이 점수를 넘는지 비교해 주세요.

| 규칙 (A 폴드, 경계 후보 .1~.99) | fold1 | fold2 | fold3 | fold4 | 평균 |
|---|---:|---:|---:|---:|---:|
| 원래 갭 (재은님 기준선) | 0.423 | 0.404 | 0.391 | 0.511 | 0.433 |
| gap_z | 0.433 | 0.431 | 0.401 | 0.512 | 0.444 |
| **gap_z + ext_range_z** | **0.434** | **0.436** | **0.402** | **0.515** | **0.447** |

## 쓰는 법

### 피처만 (모델에 넣을 때)
```python
import sys
sys.path.insert(0, "work/b/features")            # 패키지가 아니라서 경로로 추가
from gapz_rule import build_gapz
from lab.cv import feature_table

t = feature_table(days, build_gapz, name="gapz_rule")   # date, symbol, gap_z, ext_range_z, label, ret_pct
```
`build_gapz(day)` 는 `symbol, gap_z, ext_range_z` 를 돌려줌. `day.*` 로만 읽어서 cutoff(09:30) 전 데이터만 씀.

### 규칙 그대로 (모델 없이)
```python
import gapz_rule as G

params = G.fit(train_days)              # {"lam", "med", "sd", "a", "b", "train_score"}
G.save(params, "artifacts/gapz_rule.json")

pred = G.predict(day, G.load("artifacts/gapz_rule.json"))   # symbol, label (0~4), day.symbols 전부
```
`src/model.py` 에 붙일 때의 `Model` / `load_model()` 예시는 `gapz_rule.py` 맨 아래 주석에 있음.

### 확인
```bash
uv run python work/b/features/check_gapz_rule.py    # A 폴드 4폴드 평균 0.4469 재현
uv run python work/b/features/check_rules.py        # 팀 피처 규칙 1~8
```

## 그 밖의 폴더

| 폴더 | 내용 |
|---|---|
| [`eda/`](eda/) | `01~03_b_eda*.ipynb`, `figs/`(발표용 그림 01~10), `REPORT_b_eda.md`, `score_compare.py`(그림 10) |
| [`exp/`](exp/) | 피처 실험 r01 ~ r12, 공용 코드 `common.py` · `builders.py`, 결과 기록 `results.md` (목록은 `exp/README.md`) |
| [`early/`](early/) | `b_baseline.py`(첫 베이스라인 비교), `v1_check.py`(V1 26개 점검), `build_cache.py`, `test_leak.py`, `test_paste.py`, `feature_candidates.md`(실험 전 후보 목록). `work/common/folds.py`(B 홀드아웃) 기준 |
| `cache/` | 실험 중간 결과 (git 에 안 올림, 다시 만들 수 있음) |
