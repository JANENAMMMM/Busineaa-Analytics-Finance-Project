"""채민 LLM 피처 2차: 방향 대신 '확실도·크기' 로 쓰기 + 급등락 반대 문제 (2026-10-07 밤).

    uv run python scripts/a_llm2.py run [이름 ...]   # W&B tags ["llm2", llm_news_v6/llm_reddit_v4, 방식]
    uv run python scripts/a_llm2.py extra            # 채택 판정 + bootstrap + 가짜 피처 + 기간별 신호(contamination) + SHAP

근거 (강의자료 + 1차 결과. 격자는 실행 전 고정)
    w5 p25/27/39  텍스트는 방향보다 크기(변동성)를 예측. 의견 불일치(dispersion)가 움직임을 예측
    w5 p22        tone 말고 uncertainty(hedging), named event 도 텍스트에서 재는 양
    w2 p25/30     같은 확률도 경계(threshold)가 결정을 바꿈. 비용이 비대칭이면 경계도 비대칭
    w5 p35-37     LLM contamination: 모델이 기간을 기억하면 신호가 기간에 따라 달라짐 → 기간별로 비교
    w3 p37, w6    그룹 ablation, SHAP 기여도
    1차 결과      LLM 방향을 그대로 쓰면 train 이 끔(β<0, 게이트 off). 그래서 방향을 '갭과 맞나' 라는 확실도로 바꿈
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import a_llm_news as N  # noqa: E402
from lab.cv import CACHE, run_cv  # noqa: E402
from lab.folds import HOLDOUT  # noqa: E402
from src import Dataset, score  # noqa: E402
from work.a import exp2 as E  # noqa: E402
from work.a import lgb as L  # noqa: E402
from work.a.features_b import B_FEATURES_V1  # noqa: E402
from work.a.models2 import fit_rule  # noqa: E402

OUT = CACHE / "llm2"
TAGS = ["llm2"]

# ================================================================ 고정 (실행 전에 정함)
GATE_TAUS = [0.0, 0.25, 0.5]                 # 불확실 게이트: 비율 > τ 이면 0·4 를 낮춤
PLACEBO_N, PLACEBO_SEED = 30, 0
QS = E.QS                                    # 비대칭 경계 후보 분위 (B 규칙과 같음)

EXPS = {   # 이름: (방식, 피처, 출처 태그)
    # A. 크기: 사건 종류·의견 불일치 → 급등락 경계 (w5 p22/25/27)
    "llm2_rsize_news_earnguide": ("rsize", "x_news_earnguide", ["llm_news_v6"]),
    "llm2_rsize_reddit_disagree": ("rsize", "x_reddit_disagree", ["llm_reddit_v4"]),
    # B. 불확실하면 급으로 안 찍기 (회의 방향 + w5 p22 uncertainty)
    "llm2_gate_uncertain_news": ("ugate", "x_news_uncertain", ["llm_news_v6"]),
    "llm2_gate_uncertain_reddit": ("ugate", "x_reddit_uncertain", ["llm_reddit_v4"]),
    # C. 확실도 = LLM 방향이 갭과 같은 쪽인가 (같으면 +, 반대면 −) → 급등락 경계
    "llm2_rsize_agree_news": ("rsize", "x_news_agree", ["llm_news_v6"]),
    "llm2_rsize_agree_reddit": ("rsize", "x_reddit_agree", ["llm_reddit_v4"]),
    "llm2_rsize_agree_both": ("rsize", "x_both_agree", ["llm_news_v6", "llm_reddit_v4"]),
    # D. 급등락 반대 문제: 급상승·급하락 경계를 따로 (w2 p25/30, LLM 없음)
    "llm2_asym": ("asym", None, []),
    # E. 채민 피처 전부 (뉴스 26 + Reddit 20) LightGBM, 결측 그대로
    "llm2_lgb_all": ("lgb", "ALL", ["llm_news_v6", "llm_reddit_v4"]),
}
PLACEBO_EXPS = ["llm2_rsize_news_earnguide", "llm2_rsize_reddit_disagree", "llm2_gate_uncertain_news",
                "llm2_gate_uncertain_reddit", "llm2_rsize_agree_news", "llm2_rsize_agree_reddit",
                "llm2_rsize_agree_both"]


# ================================================================ 데이터

def all_llm():
    """채민 feature_* 전부, 접두어 news_f_ / reddit_f_. symbol + target 한 줄."""
    out = None
    for src, pre in [("news", "news_f_"), ("reddit", "reddit_f_")]:
        n = pd.read_csv(N.SOURCES[src]["dir"] / "daily_features.csv", encoding="utf-8-sig", parse_dates=["target"])
        assert n.target.max() < pd.Timestamp(HOLDOUT)
        cols = [c for c in n if c.startswith("feature_")]
        x = n[["symbol", "target", *cols]].rename(columns={c: pre + c[8:] for c in cols})
        out = x if out is None else out.merge(x, on=["symbol", "target"], how="outer", validate="one_to_one")
    return out


ALL_COLS = None


def build(ds):
    global ALL_COLS
    m = N.build_master(ds)
    a = all_llm()
    ALL_COLS = [c for c in a if c not in ("symbol", "target")]
    m = m.drop(columns=[c for c in ALL_COLS if c in m]).merge(a, on=["symbol", "target"], how="left",
                                                              validate="one_to_one")
    sg = np.sign(m.gap_z)
    m["x_news_earnguide"] = m.news_f_event_earnings_share + m.news_f_event_company_guidance_share
    m["x_reddit_disagree"] = m.reddit_f_stance_mixed_ratio + np.minimum(m.reddit_f_stance_bullish_ratio,
                                                                       m.reddit_f_stance_bearish_ratio)
    m["x_news_uncertain"] = (m.news_f_state_proposed_possible_share + m.news_f_state_alleged_share
                             + m.news_f_state_pending_share)
    m["x_reddit_uncertain"] = m.reddit_f_uncertainty_explicit_ratio
    m["x_news_agree"] = m.news_llm_dir * sg            # +: 뉴스가 갭과 같은 방향, −: 반대. 결측 그대로
    m["x_reddit_agree"] = m.reddit_llm_dir * sg
    m["x_both_agree"] = m[["x_news_agree", "x_reddit_agree"]].mean(axis=1, skipna=True)   # 둘 다 없으면 결측
    # 집계 규칙 2 (w5-1 p16-17): 비율 대신 부호만. 뉴스 85% 가 기사 1~2건이라 비율이 0/1 로 쏠림(민영 01)
    m["x_both_agree_sign"] = pd.concat([np.sign(m.x_news_agree), np.sign(m.x_reddit_agree)], axis=1).mean(axis=1, skipna=True)
    # dev 실적 358건 중 88% 가 예상 상회 → above 는 평범. 방향 정보는 below 쪽 (민영 01 4절)
    neg = -(m.news_llm_below.where(m.news_llm_below > 0) * sg)
    negr = -(m.reddit_llm_bear.where(m.reddit_llm_bear > 0) * sg)
    m["x_both_agree_neg"] = pd.concat([neg, negr], axis=1).mean(axis=1, skipna=True)
    E._master = m
    return m


# ================================================================ 방식

def fit_ugate(t, f):
    """B 규칙 등급 → 불확실 비율 > τ 인 0·4 를 한 칸(→1·3) 또는 보합(→2). off 포함, train 이 고름."""
    s1, (a, b), i1 = fit_rule(t, [])
    y = t.label.to_numpy(int)

    def base(x):
        return E.R.cut(N._rule_v(s1, x), a, b)

    def gate(o, u, mode, tau):
        o = o.copy()
        m = np.isfinite(u) & (u > tau) & np.isin(o, [0, 4])
        if mode == "one_step":
            o[m & (o == 0)], o[m & (o == 4)] = 1, 3
        elif mode == "to_flat":
            o[m] = 2
        return o
    o, u = base(t), t[f].to_numpy(float)
    cands = [("off", 0.0)] + [(md, tau) for md in ("one_step", "to_flat") for tau in GATE_TAUS]
    sc, mode, tau = max(((score(y, gate(o, u, md, tau))["score"], md, tau) for md, tau in cands), key=lambda r: r[0])
    return (lambda x: gate(base(x), x[f].to_numpy(float), mode, tau)), {
        "lam": i1["lam"], "gate_mode": mode, "gate_tau": tau, "train_score": sc,
        "train_changed_rows": int((gate(o, u, mode, tau) != o).sum())}


def _cut_asym(s, a, bu, bd):
    o = np.full(len(s), 2)
    o[s >= a], o[s <= -a] = 3, 1
    o[s >= bu], o[s <= -bd] = 4, 0
    o[np.isnan(s)] = 2
    return o


def fit_asym(t, f=None):
    """B 규칙 점수 s 그대로, 급상승 경계 bu 와 급하락 경계 bd 를 따로 (보합 경계 a 는 공통). 후보는 |s| 분위."""
    s1, _, i1 = fit_rule(t, [])
    v = N._rule_v(s1, t)
    y = t.label.to_numpy(int)
    q = np.quantile(np.abs(v[~np.isnan(v)]), QS)
    sc, a, bu, bd = max(((score(y, _cut_asym(v, a, bu, bd))["score"], a, bu, bd)
                         for i, a in enumerate(q) for bu in q[i + 1:] for bd in q[i + 1:]), key=lambda r: r[0])
    return (lambda x: _cut_asym(N._rule_v(s1, x), a, bu, bd)), {"lam": i1["lam"], "a": a, "b_up": bu, "b_down": bd,
                                                                "train_score": sc}


_lgb_ready = False


def make_fit(kind, f):
    global _lgb_ready
    if kind == "lgb":
        if not _lgb_ready:
            c = pd.read_parquet(L.TEXT_TABLE).drop(columns=["date", "cutoff"])
            L._text = c.merge(all_llm(), on=["symbol", "target"], how="left", validate="one_to_one")
            _lgb_ready = True
        return L.make_fit("aggr", B_FEATURES_V1 + ALL_COLS)

    def fit(train_days, exclude):
        t = E.train_rows(train_days, exclude)
        if kind == "agree_asym":
            labels, info = fit_agree_asym(t, f)
        elif kind == "ugate":
            labels, info = fit_ugate(t, f)
        elif kind == "asym":
            labels, info = fit_asym(t)
        else:
            labels, info = N.FITS[kind](t, f)
        return E.TableModel(labels, info)
    fit.__qualname__ = f"fit_llm2_{kind}"
    return fit


# ================================================================ run / extra

def run(ds, names):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.json"
    allres = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for nm in names:
        kind, f, tags = EXPS[nm]
        t0, infos, models = time.time(), {}, {}

        def fit(train_days, exclude, _f=make_fit(kind, f)):
            mdl = _f(train_days, exclude)
            infos[str(train_days[-1].date.date())] = getattr(mdl, "info", {})
            models[str(train_days[-1].date.date())] = mdl
            return mdl
        fit.__qualname__ = f"fit_llm2_{kind}"
        res = run_cv(fit, ds, name=nm, method=N.METHOD, tags=TAGS + tags + [kind], verbose=False,
                     config={"model": kind, "feature": f, "gate_taus": GATE_TAUS, "deltas": N.DELTAS,
                             "n_llm_features": len(ALL_COLS) if kind == "lgb" else None,
                             "llm_model": "gemini-3.1-flash-lite", "main_metric": "fold4/all/score"})
        a, u = N._summ(res)
        res.preds.to_parquet(OUT / f"preds_{nm}.parquet", index=False)
        if kind == "lgb":     # SHAP 용 fold4 모델 저장 (w6)
            last = sorted(models)[-1]
            models[last].booster.save_model(str(OUT / f"{nm}_fold4.txt"))
        allres[nm] = {"all": a.to_dict(), "unseen": u.to_dict(), "mean": float(a.mean()),
                      "params": json.loads(json.dumps(infos, default=float)), "seconds": round(time.time() - t0)}
        print(f"{nm:28s} " + " ".join(f"{v:.4f}" for v in a) + f" | mean {a.mean():.4f} | unseen4 {u['fold4']:.4f}"
              f" ({time.time() - t0:.0f}s)", flush=True)
        path.write_text(json.dumps(allres, indent=2), encoding="utf-8")


def contamination(m):
    """w5 p35-37: LLM 이 기간을 기억하면 신호가 기간마다 달라짐. 반기별로 '갭과 엇갈릴 때 LLM 쪽으로 간 비율' 과
    '하회/하락 의견이 있을 때 평균 수익률' 을 셈. 대상일 기준, 홀드아웃 전까지."""
    out = {}
    h = m.target.dt.year.astype(str) + "H" + np.where(m.target.dt.month <= 6, "1", "2")
    for src in ("news", "reddit"):
        rows = {}
        for per, g in m.groupby(h):
            d = g[(g[f"{src}_llm_dir"] != 0) & g[f"{src}_llm_dir"].notna() & (g.gap_z != 0) & g.gap_z.notna()]
            dis = d[np.sign(d[f"{src}_llm_dir"]) != np.sign(d.gap_z)]
            neg = g[g[N.SOURCES[src]["neg"]] > 0]
            rows[per] = {"dir_rows": len(d), "disagree": len(dis),
                         "disagree_llm_right": float((np.sign(dis.ret_pct) == np.sign(dis[f"{src}_llm_dir"])).mean()) if len(dis) else None,
                         "neg_rows": len(neg), "neg_mean_ret": float(neg.ret_pct.mean()) if len(neg) else None}
        out[src] = rows
    return out


def shap_fold4(ds, m):
    import lightgbm as lgb
    p = OUT / "llm2_lgb_all_fold4.txt"
    if not p.exists():
        return None
    bst = lgb.Booster(model_file=str(p))
    feats = bst.feature_name()
    v4 = m[m.date.isin({d.date for d in N.load_folds(ds)[-1].val})]
    contrib = bst.predict(v4[feats], pred_contrib=True)        # (n, (k+1)*5) 다항
    k = len(feats)
    c = np.abs(contrib.reshape(len(v4), 5, k + 1)[:, :, :k]).sum(axis=1).mean(axis=0)
    s = pd.Series(c / c.sum(), index=feats)
    grp = {"B26": s[[f for f in feats if f in B_FEATURES_V1]].sum(),
           "news_llm": s[[f for f in feats if f.startswith("news_f_")]].sum(),
           "reddit_llm": s[[f for f in feats if f.startswith("reddit_f_")]].sum()}
    return {"group_share": {k_: float(v) for k_, v in grp.items()},
            "top10": {k_: float(v) for k_, v in s.sort_values(ascending=False).head(10).items()},
            "top_llm": {k_: float(v) for k_, v in s[[f for f in feats if f.startswith(("news_f_", "reddit_f_"))]]
                        .sort_values(ascending=False).head(8).items()}}


def extra(ds):
    m = E._master
    runs = json.loads((OUT / "runs.json").read_text(encoding="utf-8"))
    base_runs = json.loads((N.OUT / "runs.json").read_text(encoding="utf-8"))
    r = {"adopt": {}, "boot": {}, "placebo": {}}
    for nm, v in runs.items():
        ref = "llm6_d0_lgb_aggr" if EXPS[nm][0] == "lgb" else "llm6_base"
        b = base_runs[ref]
        d = {fo: v["all"][fo] - b["all"][fo] for fo in b["all"]}
        r["adopt"][nm] = {"ref": ref, "diff": d, "mean_diff": float(np.mean(list(d.values()))),
                          "up_folds": int(sum(x > 0 for x in d.values())),
                          "pass": bool(d["fold4"] >= 0 and sum(x > 0 for x in d.values()) >= 3
                                       and np.mean(list(d.values())) >= 0.005)}
        pa, pb = pd.read_parquet(OUT / f"preds_{nm}.parquet"), pd.read_parquet(N.OUT / f"preds_{ref}.parquet")
        rng = np.random.default_rng(N.BOOT_SEED)
        r["boot"][nm] = {}
        for fo in ["fold4", "all4"]:
            a_ = pa if fo == "all4" else pa[pa.fold == fo]
            b_ = pb if fo == "all4" else pb[pb.fold == fo]
            dd = N._boot_diff(a_, b_, rng)
            r["boot"][nm][fo] = {"lo": float(np.quantile(dd, .025)), "hi": float(np.quantile(dd, .975)),
                                 "p_le0": float((dd <= 0).mean())}
    r["contamination"] = contamination(m)
    r["shap_fold4"] = shap_fold4(ds, m)
    (OUT / "extra.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    for nm in PLACEBO_EXPS:
        kind, f, _ = EXPS[nm]
        prng = np.random.default_rng(PLACEBO_SEED)
        f4, mean = [], []
        for _ in range(PLACEBO_N):
            fake = m.copy()
            ok = fake[f].notna().to_numpy()
            fake.loc[ok, f] = prng.permutation(fake.loc[ok, f].to_numpy())
            E._master = fake
            a, _ = N._summ(run_cv(make_fit(kind, f), ds, track=False, verbose=False))
            f4.append(float(a["fold4"]))
            mean.append(float(a.mean()))
        E._master = m
        r["placebo"][nm] = {"real_fold4": runs[nm]["all"]["fold4"], "real_mean": runs[nm]["mean"],
                            "fake_fold4": f4, "fake_mean": mean,
                            "fake_fold4_ge_real": int(sum(x >= runs[nm]["all"]["fold4"] for x in f4)),
                            "fake_mean_ge_real": int(sum(x >= runs[nm]["mean"] for x in mean))}
        (OUT / "extra.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    print("done")




# ================================================================ 3차 (2026-10-08 새벽, 격자 실행 전 고정)
# 근거
#  (1) llm2_rsize_agree_both 가 가짜 피처 30/30 통과, 4폴드 bootstrap 95% = +0.0004~+0.0038, p(≤0)=0.003 → 진짜 신호
#  (2) w2 p25/30 비대칭 비용 → 급상승·급하락 경계 분리 (llm2_asym fold4 +0.0038)
#  (3) 민영 01 문서 / 강의 w5-1 p16-17 "집계 규칙을 2개 이상 보고" → 비율판 외에 부호판(이진)도 같이
#  (4) 민영 01 문서 4절: dev 실적 358건 중 88% 가 예상 상회 → above 는 평범. 방향 정보는 below 쪽
#      → 'below 만' 쓰는 확실도도 따로 (뉴스 하회 / Reddit 하락 의견이 갭과 같은 쪽인가)

def fit_agree_asym(t, f):
    """급상승 경계 bu·급하락 경계 bd 를 따로 두고, 둘 다 확실도로 이동: bu·exp(−δ·x̃), bd·exp(−δ·x̃)."""
    s1, _, i1 = fit_rule(t, [])
    ms = N._std_obs(t[f])
    v, y = N._rule_v(s1, t), t.label.to_numpy(int)

    def xt(x):
        return np.clip((np.asarray(x[f], float) - ms[0]) / ms[1], -5, 5)

    def lab(v, xf, a, bu, bd, d):
        o = np.full(len(v), 2)
        o[v >= a], o[v <= -a] = 3, 1
        e = np.where(np.isnan(xf), 1.0, np.exp(-d * np.nan_to_num(xf)))
        o[v >= bu * e], o[v <= -bd * e] = 4, 0
        o[np.isnan(v)] = 2
        return o
    xf = xt(t)
    q = np.quantile(np.abs(v[~np.isnan(v)]), QS)
    sc, a, bu, bd, d = max(((score(y, lab(v, xf, a, bu, bd, d))["score"], a, bu, bd, d)
                            for i, a in enumerate(q) for bu in q[i + 1:] for bd in q[i + 1:] for d in N.DELTAS),
                           key=lambda r: r[0])
    return (lambda x: lab(N._rule_v(s1, x), xt(x), a, bu, bd, d)), {
        "lam": i1["lam"], "a": a, "b_up": bu, "b_down": bd, "delta": d, "train_score": sc}


EXPS["llm2_agree_asym"] = ("agree_asym", "x_both_agree", ["llm_news_v6", "llm_reddit_v4"])
EXPS["llm2_rsize_agree_sign"] = ("rsize", "x_both_agree_sign", ["llm_news_v6", "llm_reddit_v4"])
EXPS["llm2_rsize_agree_neg"] = ("rsize", "x_both_agree_neg", ["llm_news_v6", "llm_reddit_v4"])
PLACEBO_EXPS += ["llm2_agree_asym", "llm2_rsize_agree_sign", "llm2_rsize_agree_neg"]


if __name__ == "__main__":
    cmd = sys.argv[1]
    ds = Dataset()
    build(ds)
    if cmd == "run":
        run(ds, sys.argv[2:] or list(EXPS))
    elif cmd == "extra":
        extra(ds)
