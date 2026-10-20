"""팀 W&B 에 B0 와 분봉 후보를 run_cv 로 올림 (16 문서 A안: 채택 판단에 필요한 것만).

    uv run python work/b/review/premarket/log_wandb.py

run 4개 (태그 b_review, premarket)
    b0_gapz          지금 제출 규칙 (교수님 08:00 봉 갭)
    premarket_0915   후보: 갭 가격 = 9:15 확정 1분봉 (무료 SIP 15분 지연 감안, 판정안)
    premarket_0900   참고: 9:00 확정
    premarket_0929   참고: 9:29 확정 (실시간 SIP 유료가 있어야 쓸 수 있음)

- 분봉 저장소는 cache/premarket/ 원본 1분봉을 premarket_rule.reduce_bars 로 시점별로 줄여서 만듦 (cache/premarket_XXXX/)
- 학습은 fold.train 에서 unseen 종목을 뺀 행, 예측은 run_cv 가 guard(day) 로 부름
- 올라가는 것: 점수·혼동행렬·설정. 원본 데이터·분봉·키는 안 올라감
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work/b/features"))

from lab.cv import feature_table, run_cv  # noqa: E402
import gapz_rule as G  # noqa: E402
import premarket_rule as PR  # noqa: E402

RAW = ROOT / "cache" / "premarket"
CUTS = {"0900": (9, 0), "0915": (9, 15), "0929": (9, 29)}


def make_store(key):
    hh, mm = CUTS[key]
    store = ROOT / "cache" / f"premarket_{key}"
    if key == "0915":
        store = ROOT / "cache" / "premarket_0915"          # check_premarket_rule.py 와 같은 저장소
    if store.exists() and len(list(store.glob("*.parquet"))) >= 449:
        return store
    store.mkdir(parents=True, exist_ok=True)
    as_of = pd.Timedelta(hours=hh, minutes=mm)
    for f in sorted(RAW.glob("*.parquet")):
        PR.reduce_bars(pd.read_parquet(f), f.stem, as_of).to_parquet(store / f.name, index=False)
    return store


class RuleModel:
    def __init__(self, params, build):
        self.params, self.build = params, build

    def predict(self, day):
        x = self.build(day)
        out = pd.DataFrame({"symbol": day.symbols})
        if len(x):
            x = x.assign(label=G.predict_table(x, self.params))
            out = out.merge(x[["symbol", "label"]], on="symbol", how="left")
        else:
            out["label"] = 2
        out["label"] = out.label.fillna(2).astype(int)
        return out


def make_fit(build):
    """피처 표를 날짜별로 한 번 만들어 두고, 폴드마다 train 날짜·종목만 골라 규칙을 학습."""
    cache = {}

    def fit(train_days, exclude):
        need = [d for d in train_days if d.date not in cache]
        if need:
            t = feature_table(need, build, name=None, verbose=False)
            for dt, g in t.groupby("date"):
                cache[pd.Timestamp(dt)] = g
        tr = pd.concat([cache[d.date] for d in train_days if d.date in cache], ignore_index=True)
        tr = tr[~tr.symbol.isin(exclude)].copy()
        tr["label"] = tr.label.astype(int)
        return RuleModel(G.fit_table(tr), build)
    return fit


def main():
    runs = [("b0_gapz", G.build_gapz, {"gap_price": "교수님 08:00 봉 종가 (실제 약 8:30 가격)", "as_of": "09:00 공개"})]
    for key in ["0915", "0900", "0929"]:
        store = make_store(key)
        hh, mm = CUTS[key]
        note = {"0915": "판정안 (무료 SIP 15분 지연 → 9:30 전에 받을 수 있음)",
                "0900": "참고", "0929": "참고 (실시간 SIP 유료가 있어야 사용 가능)"}[key]
        runs.append((f"premarket_{key}", lambda day, s=store: PR.build_gapz_pm(day, s),
                     {"gap_price": f"Alpaca SIP 1분봉, 봉 끝 ≤ {hh:02d}:{mm:02d}", "as_of": f"{hh:02d}:{mm:02d}",
                      "note": note, "fallback": "분봉 없거나 단위 2% 초과 어긋나면 B0 갭"}))
    for name, build, cfg in runs:
        print(f"\n=== {name}", flush=True)
        res = run_cv(make_fit(build), name=name, tags=["b_review", "premarket"],
                     config={"rule": "gapz_rule (gap_z + ext_range_z, λ·a·b train)", "doc": "docs/review/16_프리마켓_분봉.md", **cfg})
        s = res.table[res.table.group == "all"].score
        u = res.table[res.table.group == "unseen"].score
        print(f"  평균 {s.mean():.4f}, unseen {u.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
