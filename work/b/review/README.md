# B 리뷰 스크립트·그림 (민영)

문서는 [docs/review/](../../../docs/review/README.md) 에 있습니다. 이 폴더에는 문서가 쓰는 분석 스크립트와 그림만 둡니다.
결과 CSV 는 git 에 안 올리고, 스크립트를 돌리면 `out/` 에 다시 생깁니다.

| 폴더 | 스크립트 | 쓰는 문서 |
|---|---|---|
| [explain/](explain/) | `explain_rule.py` 예측 분해, permutation, 그룹별 효과 | 03 |
| [news_gap/](news_gap/) | `news_gap.py` 뉴스 있는 갭, 시장 전체가 뒤집힌 날, 뉴스 수 baseline | 05 |
| [batch/](batch/) | `batch.py` 배치 실험 31개 | 06 |
| [rest/](rest/) | `rest.py` LightGBM 묶음 중요도·faithfulness, 실적×갭, 처음 보는 종목, 텍스트 항목 (nogap 이 이 파일의 LightGBM 절차를 씀) | 07 · 08 |
| [nogap/](nogap/) | `nogap.py` 갭 없이 금융·텍스트 구성 8가지 | 08 |
| [errors/](errors/) | `errors.py` 오답 분석 | 09 |
| [tweak/](tweak/) | `tweak.py` 오답을 겨냥한 규칙 수정 | 10 |
| [more/](more/) | `more.py` 규칙 구조 실험 | 11 |
| [papers/](papers/) | `papers.py` 논문 기반 실험 A·B·C | 13 |
