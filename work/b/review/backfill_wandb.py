"""B 리뷰 실험(docs/review 05~13, 프리마켓 16 은 log_wandb*.py) 결과를 팀 W&B 에 소급 업로드 — 실험마다 run 하나.

    uv run python work/b/review/backfill_wandb.py          # 미리 보기 (올릴 목록만)
    uv run python work/b/review/backfill_wandb.py --go     # 업로드

- 다시 돌리지 않음. 실험 당시 저장한 결과 CSV 의 숫자를 그대로 올림 (work/b/cache/*.csv, work/b/review/*/out/*.csv).
- 제외: 프리마켓 실험 (log_wandb*.py 로 이미 27개), 재은이 올린 B 규칙 기준선과 같은 행 (rule_gapz, rule_gapz_ext).
- 키 이름은 팀 run_cv 와 같게: fold{k}/all/score, mean/all/score, mean/unseen/score (+ delta_vs_base). 차이(Δ)만 저장된 결과는
  기준선 점수에 더해 점수로 되돌리고, 어떤 기준선인지 config.base 에 적음.
- 태그: b_review, backfill, orig:<원래 날짜>, 묶음 이름, adopted / not_adopted. job_type = backfill.
- 큰 격자(7차 225개, 9차 39개, 11차 42개, 단독 규칙 90개, 12차 조합)는 상위 변형만 (config.selection 에 기준).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
C = HERE.parent / "cache"
R = HERE
F = ["fold1", "fold2", "fold3", "fold4"]

# 기준선 (폴드 순)
B0 = [0.434422, 0.435603, 0.402021, 0.515419]          # gap_z + ext_range_z, 경계 후보 .1~ (재은 rule_gapz_ext, 제출 v1)
GZ5 = [0.4404, 0.4153, 0.4021, 0.4960]                  # gap_z 규칙, 경계 후보 .5~ (B 6~12차 실험의 기준)
DOC = {"exp": "work/b/exp/results.md", "05": "docs/review/05_뉴스_시장_실험.md", "06": "docs/review/06_배치_실험.md",
       "07": "docs/review/07_남은_항목.md", "08": "docs/review/08_갭_없이.md", "10": "docs/review/10_규칙_수정.md",
       "11": "docs/review/11_규칙_구조_실험.md", "13": "docs/review/13_논문_기반_실험.md", "early": "work/b/eda/REPORT_b_eda.md"}

rows = []


def add(family, variant, date, doc, folds=None, unseen=None, adopted=False, base=None, delta=None, extra=None,
        method="A-folds4", selection=None, metrics=None):
    rows.append({"family": family, "variant": str(variant), "date": date, "doc": DOC[doc], "folds": folds, "unseen": unseen,
                 "adopted": bool(adopted), "base": base, "delta": delta, "extra": extra or {}, "method": method,
                 "selection": selection, "metrics": metrics or {}})


def from_delta(d, base):
    return [b + x for b, x in zip(base, d)]


def num(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def rd(f):
    return pd.read_csv(f, encoding="utf-8-sig")


# ------------------------------------------------------------------ 초기 (B 홀드아웃: val 2026-03~05 월별)
def early():
    b = rd(C / "baseline_results.csv")
    for m, g in b.groupby("model", sort=False):
        g = g.set_index("fold")
        add("early_baseline", m, "2026-10-01", "early", method="B-holdout (val 2026-03~05, 월별)",
            metrics={f"month/{k}": v for k, v in g.score.items()} | {"mean/all/score": g.score.mean()})
    for fname, fam in [("v1_diag.csv", "early_v1_diag"), ("v1_universe.csv", "early_v1_universe")]:
        d = rd(C / fname)
        for m, g in d.groupby("model", sort=False):
            g = g.groupby("fold").score.mean()
            add(fam, m, "2026-10-01", "early", method="B-holdout (val 2026-03~05, 월별)",
                metrics={f"month/{k}": v for k, v in g.items()} | {"mean/all/score": g.mean()})
    d = rd(C / "v1_holdout.csv")
    for m, g in d.groupby("model", sort=False):
        g = g.groupby("fold").score.mean()
        add("early_v1_unseen10", m, "2026-10-01", "early", method="B-holdout, 10종목 학습 제외 (시드 평균)",
            metrics={f"month/{k}": v for k, v in g.items()} | {"mean/all/score": g.mean()})


# ------------------------------------------------------------------ B 피처 실험 1~12차 (A 폴드)
def seeds_table(fname, fam, keycol="variant", label=None):
    d = rd(C / fname)
    for v, g in d.groupby(keycol, sort=False):
        m = g.groupby("fold")[["all", "unseen"]].mean()
        add(fam, v, "2026-10-02", "exp", folds=m["all"].reindex(F).tolist(), unseen=m["unseen"].mean(),
            extra={"model": label or "LightGBM 다중분류 (시드 3개 평균)"})


def wide(fname, fam, keycols, label, base=None, skip=(), top=None, sort_col="4폴드 평균"):
    d = rd(C / fname)
    if top:
        d = d.sort_values(sort_col, ascending=False).head(top)
    for _, r in d.iterrows():
        v = " | ".join(f"{k}={r[k]}" for k in keycols if k in r and pd.notna(r[k])) if len(keycols) > 1 else r[keycols[0]]
        if str(v) in skip:
            continue
        add(fam, v, "2026-10-02", "exp", folds=[r[f] for f in F], unseen=num(r.get("unseen")), base=base,
            extra={"model": label}, selection=f"{sort_col} 상위 {top}" if top else None)


def delta_rows(fname, fam, keycols, base, label, top, sort_col="Δ평균", fold_cols=None, adopted_col="채택"):
    d = rd(C / fname).sort_values(sort_col, ascending=False).head(top)
    for _, r in d.iterrows():
        v = " · ".join(str(r[k]) for k in keycols)
        if fold_cols:
            dl = [r[c] for c in fold_cols]
            folds = from_delta(dl, base)
        else:
            dl, folds = None, None
        add(fam, v, "2026-10-02", "exp", folds=folds, base=base, delta=r[sort_col], adopted=bool(r.get(adopted_col, False)),
            extra={"model": label, "β": str(r.get("β", ""))}, selection=f"{sort_col} 상위 {top} (전체 {len(rd(C / fname))}개)",
            metrics={} if fold_cols else {"delta_vs_base/mean": r[sort_col], "delta_vs_base/fold4": num(r.get("Δfold4")),
                                          "up_folds": num(r.get("상승폴드"))})


def exp_rounds():
    seeds_table("exp_ablation.csv", "exp01_ablation")                                    # 1차
    seeds_table("exp_round2_DS.csv", "exp01_round2_DS")
    seeds_table("exp_round2_DS_leaf100.csv", "exp01_round2_leaf100")
    seeds_table("exp_decode_compare.csv", "exp01_decode_compare")
    seeds_table("exp_pruned.csv", "exp01_pruned")
    wide("exp_decode_summary.csv", "exp02_decode", ["kind", "K", "F"], "LightGBM V1 + 디코딩 (급등락 K, 보합 F)")  # 2차
    seeds_table("exp_decode_feats.csv", "exp02_decode_feats")
    wide("exp_twostage_summary.csv", "exp03_twostage", ["model", "K", "F"], "2단계 구조")                       # 3차
    wide("exp_gapplus.csv", "exp04_gapplus", ["model", "param"], "갭 규칙 + α (경계 후보 .5~)", base=GZ5)          # 4차
    wide("exp_gapz2_denom.csv", "exp05_denominator", ["피처"], "갭 분모 바꾸기 (경계 후보 .5~)", base=GZ5)         # 5차
    wide("exp_gapz2_blend.csv", "exp05_blend", ["방식", "w"], "gap_z 규칙 × LightGBM 섞기", base=GZ5)
    g = rd(C / "exp_gapfeat.csv")                                                        # 6차 (파일엔 macd_hist 재실행만 남음)
    for v, gg in g.groupby("피처", sort=False):
        m = gg.loc[gg.groupby("fold").train.idxmax()].set_index("fold")
        add("exp06_gapfeat", v, "2026-10-02", "exp", folds=m["all"].reindex(F).tolist(), unseen=m.unseen.mean(), base=GZ5,
            extra={"model": "gap_z + 피처 하나 (더하기, β 는 폴드마다 train 최고)", "β": "/".join(map(str, m.beta.reindex(F))),
                   "note": "6차 나머지 25개 피처는 결과 파일이 덮어써져 results.md 6차 표에만 있음"})
    delta_rows("exp_gapfeat_all_summary.csv", "exp07_gapfeat_all", ["피처", "방식"], GZ5,         # 7차
               "gap_z + 피처 × 3방식", 20, fold_cols=["Δfold1", "Δfold2", "Δfold3", "Δfold4"])
    mm = rd(C / "exp_multi.csv")                                                         # 8차
    for v, gg in mm.groupby("model", sort=False):
        if "gap_z 규칙" in v:
            continue
        m = gg.groupby("fold")[["all", "unseen"]].mean()
        add("exp08_multi", v, "2026-10-02", "exp", folds=m["all"].reindex(F).tolist(), unseen=m.unseen.mean(), base=GZ5,
            extra={"model": "여러 피처 함께 (train 내부 검증으로 고름)"})
    wide("exp_more_A.csv", "exp09_rule_shape", ["model"], "규칙 모양 바꾸기", base=GZ5, skip={"A0 gap_z 규칙"})  # 9차
    delta_rows("exp_more_BC.csv", "exp09_new_feats", ["피처", "방식"], GZ5, "새 피처 13개 × 3방식", 10)
    wide("exp_prior_summary.csv", "exp10_prior", ["model"], "미리 정한 조합", base=GZ5, skip={"gap_z 규칙"})       # 10차
    s = rd(C / "exp_solo.csv").sort_values("혼자 규칙 4폴드", ascending=False).head(15)                         # 단독
    for _, r in s.iterrows():
        add("exp10b_solo", r["피처"], "2026-10-02", "exp", extra={"model": "피처 하나만으로 규칙 (gap_z 없이)"},
            selection="혼자 규칙 4폴드 상위 15 (전체 90개)",
            metrics={"mean/all/score": r["혼자 규칙 4폴드"], "fold4/all/score": r["fold4"], "dir_ic": num(r["방향 IC"]),
                     "mag_ic": num(r["크기 IC"])})
    delta_rows("exp_raw_gapz.csv", "exp11_raw", ["피처", "방식"], GZ5, "원본 잔여 정보 14개 × 3방식", 10)          # 11차
    cb = rd(C / "exp_combo.csv")                                                         # 12차: 폴드마다 내부 최고 조합
    for k in (1, 2, 3):
        pick = cb[cb["개수"] == k].sort_values("내부Δ", ascending=False).groupby("fold").head(1).set_index("fold")
        add("exp12_combo", f"{k}개 조합 (내부 검증 최고)", "2026-10-02", "exp", folds=from_delta(pick.valΔ.reindex(F), GZ5),
            unseen=pick.unseen.mean(), base=GZ5, extra={"model": "2·3개 조합 전수 (train 내부 검증으로 고름)",
                                                         "chosen": " / ".join(f"{f}: {pick.loc[f, '조합']}" for f in F)})


# ------------------------------------------------------------------ 리뷰 실험 (docs/review)
def review():
    j = rd(R / "batch/out/judge.csv")                                                    # 06 배치
    for _, r in j.iterrows():
        add("rev06_batch", f"{r['이름']} [{r['form']}]", "2026-10-08", "06", folds=from_delta([r[f] for f in F], B0),
            base=B0, delta=r["평균"], adopted=r["채택"], extra={"group": r["group"], "param": r["파라미터"],
                                                              "test": r["test"], "fake95_pass": bool(r["가짜95초과"])})
    j = rd(R / "news_gap/out/3_judge.csv")                                               # 05 뉴스·시장
    for _, r in j.iterrows():
        add("rev05_news_gap", f"{r['feature']} [{r['form']}]", "2026-10-08", "05", folds=from_delta([r[f] for f in F], B0),
            base=B0, delta=r["평균"], adopted=r["채택"], extra={"param": r["파라미터"]})
    j = rd(R / "rest/out/01_text_cv.csv")                                                # 07 남은 항목 (텍스트)
    for _, r in j.iterrows():
        add("rev07_text", f"{r['실험']} [{r['형태']}]", "2026-10-08", "07", folds=from_delta([r[f] for f in F], B0),
            base=B0, delta=r["평균"], adopted=r["채택"], extra={"param": r["파라미터"]})
    j = rd(R / "nogap/out/summary.csv")                                                  # 08 갭 없이
    for _, r in j.iterrows():
        if "제출본" in r["config"]:
            continue
        add("rev08_nogap", r["config"].strip(), "2026-10-08", "08", folds=[r[f] for f in F], unseen=r["unseen"],
            extra={"n_feat": num(r["n_feat"]), "model": "LightGBM (갭 없이/있이 구성)"})
    j = rd(R / "tweak/out/judge.csv")                                                    # 10 규칙 수정
    for _, r in j.iterrows():
        add("rev10_tweak", r["후보"], "2026-10-08", "10", folds=from_delta([r[f] for f in F], B0), base=B0, delta=r["평균"],
            adopted=r["통과"], extra={"param": str(r["param"])[:250]})
    j = rd(R / "more/out/judge.csv")                                                     # 11 구조
    for _, r in j.iterrows():
        add("rev11_structure", r["실험"], "2026-10-09", "11", folds=from_delta([r[f] for f in F], B0), base=B0, delta=r["평균"],
            adopted=r["채택"], extra={"param": str(r["param"])[:250]})
    j = rd(R / "papers/out/papers.csv")                                                  # 13 논문 (점수 아님, 유지율 검정)
    j["exp"] = j["실험"].str[0]                                                          # A / B / C (참고 비교 포함)
    for e, g in j.groupby("exp"):
        main_name = g[~g["실험"].str.contains("참고")]["실험"].iloc[0]
        m = {}
        for _, r in g.iterrows():
            k = ("ref_" if "참고" in r["실험"] else "") + ("disc" if r["기간"] == "발견" else "conf")
            m.update({f"{k}/diff": r["차이"], f"{k}/ci95_lo": r["95% 하한"], f"{k}/ci95_hi": r["95% 상한"],
                      f"{k}/n_A": r["n A"], f"{k}/n_B": r["n B"]})
        add("rev13_papers", main_name, "2026-10-10", "13", adopted=bool(g["통과"].any()),
            method="갭 유지율 비교 (점수 run 아님). disc = 발견 기간, conf = 확인 기간, ref_ = 참고 비교", metrics=m,
            extra={"reference": g[g["실험"].str.contains("참고")]["실험"].iloc[0]})


def run_name(r):
    v = re.sub(r"[^\w가-힣.+=·|-]+", "_", r["variant"]).strip("_")
    return f"b_{r['family']}_{v}"[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    review()                    # 10/12 결정: W&B 에는 리뷰 실험(docs/review 05~13)만. 초기·B 피처 실험은 results.md 에만
    df = pd.DataFrame(rows)
    df["name"] = df.apply(run_name, axis=1)
    dup = df.name.duplicated(keep=False)
    df.loc[dup, "name"] = df.loc[dup, "name"] + "_" + df[dup].groupby("name").cumcount().astype(str)
    print(f"올릴 run {len(df)}개 (묶음 {df.family.nunique()}개)")
    print(df.groupby("family", sort=False).size().to_string())
    if not a.go:
        print("\n예시:")
        for _, r in df.sample(5, random_state=0).iterrows():
            print(" ", r["name"], r["folds"], r["metrics"])
        return

    import wandb
    cfg = json.loads((ROOT / "lab" / "wandb.json").read_text(encoding="utf-8"))
    done = {r.name for r in wandb.Api(timeout=90).runs(f"{cfg['entity']}/{cfg['project']}", per_page=300)}
    for i, r in df.iterrows():
        if r["name"] in done:
            print("  이미 있음, 건너뜀:", r["name"])
            continue
        tags = ["b_review", "backfill", f"orig:{r['date']}", r["family"], "adopted" if r["adopted"] else "not_adopted"]
        config = {"family": r["family"], "variant": r["variant"], "original_date": r["date"], "doc": r["doc"],
                  "method": r["method"], "source": "실험 당시 결과 CSV (다시 돌리지 않음)", **r["extra"]}
        if r["base"] is not None:
            config["base"] = "B0 rule_gapz_ext (0.4469)" if r["base"] == B0 else "gap_z 규칙, 경계 후보 .5~ (0.4384)"
        if r["selection"]:
            config["selection"] = r["selection"]
        run = wandb.init(entity=cfg["entity"], project=cfg["project"], name=r["name"], tags=tags, config=config,
                         job_type="backfill", reinit=True, settings=wandb.Settings(silent=True, console="off"))
        s = {}
        if r["folds"] is not None and all(x is not None and not np.isnan(x) for x in r["folds"]):
            for f, v in zip(F, r["folds"]):
                s[f"{f}/all/score"] = float(v)
            s["mean/all/score"] = float(np.mean(r["folds"]))
            s["std/all/score"] = float(np.std(r["folds"], ddof=1))
            if r["base"] is not None:
                s["delta_vs_base/mean"] = float(np.mean(r["folds"]) - np.mean(r["base"]))
                s["delta_vs_base/fold4"] = float(r["folds"][3] - r["base"][3])
                s["up_folds"] = int(sum(x > b + 1e-12 for x, b in zip(r["folds"], r["base"])))
        if r["unseen"] is not None and not pd.isna(r["unseen"]):
            s["mean/unseen/score"] = float(r["unseen"])
        s.update({k: float(v) for k, v in r["metrics"].items() if v is not None and not pd.isna(v)})
        s["adopted"] = int(r["adopted"])
        run.summary.update(s)
        run.finish()
        if i % 20 == 0:
            print(f"  {i + 1}/{len(df)} {r['name']}", flush=True)
    print("완료")


if __name__ == "__main__":
    main()
