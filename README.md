# 전주 좌표 기반 산불 위험 예측

전주(`pole`)의 좌표를 입력받아 산불 위험 여부를 예측하는 공모전 제출용 머신러닝 파이프라인입니다.

최종 제출 모델은 새 전주에도 적용할 수 있도록, 원본 GIS가 없으면 재계산할 수 없는 지형·토지피복·산림 피처를 제외하고 다음 75개 피처만 사용합니다.

- 기상 피처 32개
- 과거 화재 이력 피처 32개
- 좌표에서 재계산 가능한 등산로 피처 11개

LightGBM, XGBoost, CatBoost, RandomForest를 학습한 뒤 검증 PR-AUC에 비례해 가중 앙상블하고, 검증 데이터에서 선택한 F1 기준 임계값을 추론에 고정 적용합니다.

## 최종 파이프라인

```text
wildfire 원본 사건
        +
genfire 중 산불·들불 사건
        ↓ 날짜 및 2km 기준 중복 제거
wildfire_cleaned_merged.csv
        ↓
전주 × 날짜 skeleton 생성
        ↓
날씨 lag1 + 화재 이력 + 등산로 피처 생성
        ↓
2014~2023 rolling-origin 6-fold
        ↓
LGB + XGB + CatBoost + RandomForest
        ↓
PR-AUC 비례 가중 앙상블
        ↓
2024 holdout 평가
        ↓
새 전주 또는 test_hanjeon 추론
```

## 저장소 구조

```text
wildfire-risk-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── submission_code/
│   ├── merge_genfire_wildfire_events.py
│   ├── build_dataset_skeleton_submission_wildfire.py
│   ├── build_features_submission_wildfire.py
│   ├── build_folds_submission_wildfire.py
│   ├── train_submission_wildfire.py
│   ├── eval_utils.py
│   ├── score_new_poles.py
│   ├── score_test_hanjeon.py
│   ├── spatial_cv_check.py
│   ├── ablation_static_features.py
│   └── shap_analysis_submission_wildfire.py
├── docs/
│   ├── RESULTS.md
│   ├── DATA_AND_ARTIFACTS.md
│   ├── FILE_CLASSIFICATION.md
│   └── PLAN_full_feature_research.md
├── data/
│   └── README.md
└── archive/
    └── README.md
```

## 실행 전 경로 설정

현재 스크립트에는 다음과 같은 로컬 경로가 하드코딩되어 있습니다.

```python
BASE = Path(r"C:\Users\kim20\Downloads\Cleaned_data")
```

각 스크립트의 `BASE` 또는 `BUILD_DIR`을 실제 데이터 폴더에 맞게 수정해야 합니다.

## 실행 순서

### 1. 사건 데이터 병합

```bash
python submission_code/merge_genfire_wildfire_events.py
```

`wildfire_cleaned_filled.csv`와 `fire_clean (2).csv`의 산불·들불 사건을 병합하고 중복을 제거하여 `wildfire_cleaned_merged.csv`를 만듭니다.

### 2. 제출용 skeleton 생성

```bash
python submission_code/build_dataset_skeleton_submission_wildfire.py
```

결과:

```text
build/skeleton_wildfire_merged.parquet
```

이 스크립트는 업로드된 기존 skeleton 생성 코드를 기반으로 재현성 보완을 위해 추가한 정리본입니다. 실제 최종 실행본과 샘플링 상수가 같은지 확인한 뒤 사용해야 합니다.

### 3. 제출용 피처 생성

```bash
python submission_code/build_features_submission_wildfire.py
```

결과:

```text
build/features_full_wildfire_submission.parquet
```

### 4. Rolling-origin 폴드 생성

```bash
python submission_code/build_folds_submission_wildfire.py
```

2018~2023을 순차적으로 검증 연도로 사용하고, 2024를 최종 테스트로 분리합니다.

### 5. 모델 학습

```bash
python submission_code/train_submission_wildfire.py
```

모델과 평가 결과는 다음 폴더에 저장됩니다.

```text
build/models/wildfire_submission/
```

### 6. 추론

기존 전체 전주와 좌표가 같은 공모전 파일:

```bash
python submission_code/score_test_hanjeon.py
```

임의의 새 전주 목록:

```bash
python submission_code/score_new_poles.py new_poles.csv scored_poles.csv
```

입력 CSV에는 다음 컬럼이 필요합니다.

```text
pole_id, lon, lat
```

### 7. 선택 분석

```bash
python submission_code/shap_analysis_submission_wildfire.py
python submission_code/spatial_cv_check.py wildfire
python submission_code/ablation_static_features.py wildfire
```

## 최종 결과 요약

최종 제출 모델은 75개 피처를 사용했습니다.

| 모델 | Test AUC | Test PR-AUC |
|---|---:|---:|
| LightGBM | 0.8176 | 0.8033 |
| XGBoost | 0.8118 | 0.7972 |
| CatBoost | 0.8247 | 0.8237 |
| RandomForest | 0.8182 | 0.8043 |
| Ensemble | 0.8252 | 0.8047 |
| Calibrated Ensemble | 0.8267 | 0.8093 |

검증 PR-AUC 기반 앙상블 가중치는 다음과 같습니다.

```text
LightGBM 0.196
XGBoost  0.195
CatBoost 0.350
RF       0.258
```

`test_hanjeon.csv` 1,387,831건 추론에서는 기존 전주 좌표와 모두 매칭되었고, 임계값 `0.0005`에서 위험 판정 비율은 `31.576%`였습니다.

## 데이터 및 모델 파일

원본 CSV, Parquet, Shapefile, 학습 모델은 용량과 배포 권한 문제로 GitHub에 포함하지 않습니다. 필요한 파일과 권장 배치는 [`docs/DATA_AND_ARTIFACTS.md`](docs/DATA_AND_ARTIFACTS.md)를 참고하세요.

## 문서 주의사항

`docs/PLAN_full_feature_research.md`는 wildfire·genfire 전체 피처 연구 파이프라인의 과거 계획서입니다. 최종 제출 모델은 이 계획서와 달리 다음을 적용합니다.

- 전체 정적 GIS 피처 대신 재계산 가능한 75개 피처 사용
- 공모전 평가 기준에 맞춰 F2가 아닌 F1 임계값 사용
- `wildfire_cleaned_merged.csv` 기준 학습 및 추론

따라서 최종 실행 기준 문서는 이 README입니다.
