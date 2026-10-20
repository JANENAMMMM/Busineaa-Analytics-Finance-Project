"""채민 LLM 피처 실험: 뉴스 v6 + Reddit v4 (dev, target 2024-08-14 ~ 2026-05-29, 4폴드).

    uv run python scripts/a_llm_news.py check          # 1. ZIP 내용 검증
    uv run python scripts/a_llm_news.py run [이름 ...]  # 2. 실험 (W&B tags llm_news_v6 / llm_reddit_v4)
    uv run python scripts/a_llm_news.py extra          # 3. 표본 수·bootstrap·가짜 피처·오답 행 분석

데이터 (origin/chaemin-llm-v6, git 무시 폴더에 풂)
    cache/llm_news_v6_dev/   ← llm_news_v6_dev_handoff_final.zip
    cache/llm_reddit_v4/     ← reddit_v4_dev_handoff_final (1).zip
    cache/llm_news_v6/       ← 첫 판 (target ~2026-02-12). check 에서 dev 판과 겹치는 기간이 같은지만 봄

결합: symbol + target. run_cv 예측의 date 열 = 대상일(target), cache/b_v1.parquet 의 date = 기준일.
결측: 0 으로 채우지 않음. 피처가 없는 행은 "B 규칙 그대로" 가 되게 각 방식에서 따로 처리함.
후보 격자는 결과를 보기 전에 고정함 (2026-10-07 오전, 뉴스. Reddit 은 같은 격자를 그대로 씀).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lab.cv import CACHE, run_cv  # noqa: E402
from lab.folds import HOLDOUT, load_folds  # noqa: E402
from src import Dataset, score  # noqa: E402
from work.a import exp2 as E  # noqa: E402
from work.a import lgb as L  # noqa: E402
from work.a.features_b import B_FEATURES_V1  # noqa: E402
from work.a.models2 import fit_rule  # noqa: E402

OUT = CACHE / "llm_news_v6_dev" / "results"
METHOD = "A-folds4 (주 지표 fold4)"

# ================================================================ 고정 격자 (실행 전에 정함)

SOURCES = {   # 출처: 파일, 열 이름 바꾸기, 방향(+)/(−) 열
    "news": {"dir": CACHE / "llm_news_v6_dev", "tag": "llm_news_v6",
             "rename": {"feature_relevant_share": "news_llm_relevant",       # 관련 기사 비율 (크기 후보)
                        "feature_surprise_above_share": "news_llm_above",    # 예상 상회
                        "feature_surprise_below_share": "news_llm_below"},   # 예상 하회
             "pos": "news_llm_above", "neg": "news_llm_below"},
    "reddit": {"dir": CACHE / "llm_reddit_v4", "tag": "llm_reddit_v4",
               "rename": {"feature_relevance_direct_ratio": "reddit_llm_direct",  # 직접 관련 비율 (크기 후보)
                          "feature_stance_bullish_ratio": "reddit_llm_bull",      # 작성자 상승 의견
                          "feature_stance_bearish_ratio": "reddit_llm_bear"},     # 작성자 하락 의견
               "pos": "reddit_llm_bull", "neg": "reddit_llm_bear"},
}
# 파생: <src>_llm_dir = pos − neg (방향, −1~1), <src>_llm_surp = pos + neg (방향 있는 글 비율, 크기)

# a. 급등락 경계 이동  b' = b·exp(−δ·x̃),  x̃ = train 관측 행 기준 표준화, 결측 행은 b 그대로
DELTAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]          # models2.DELTAS 와 같음
# b. 방향 게이트: 방향이 규칙과 반대이고 |dir| > τ 인 급등락(0·4) 행을 낮춤
GATE_MODES = ["off", "one_step", "to_flat"]               # off=규칙 그대로, one_step 0→1·4→3, to_flat 0·4→2
GATE_TAUS = [0.0, 0.25, 0.5]
# c. 방향 더하기  s' = s + β·sd(s)·dir,  dir 결측이면 더하는 항 없음
BETAS = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]
# d. LightGBM 다항 (B 26개 + LLM), 디코딩 aggr (work/a/lgb.py 그대로), LightGBM 이 결측을 직접 처리
NEWS_F = ["news_llm_relevant", "news_llm_above", "news_llm_below", "news_llm_dir"]
REDDIT_F = ["reddit_llm_direct", "reddit_llm_bull", "reddit_llm_bear", "reddit_llm_dir"]

EXPS = {   # 이름: (방식, 피처, 출처 태그)
    "llm6_base": ("rule", None, ["llm_news_v6", "llm_reddit_v4"]),     # 기준: v2_rule_base
    "llm6_a_rsize_relevant": ("rsize", "news_llm_relevant", ["llm_news_v6"]),
    "llm6_a_rsize_surp": ("rsize", "news_llm_surp", ["llm_news_v6"]),
    "llm6_b_gate_dir": ("gate", "news_llm_dir", ["llm_news_v6"]),
    "llm6_c_radd_dir": ("radd", "news_llm_dir", ["llm_news_v6"]),
    "rd4_a_rsize_direct": ("rsize", "reddit_llm_direct", ["llm_reddit_v4"]),
    "rd4_a_rsize_surp": ("rsize", "reddit_llm_surp", ["llm_reddit_v4"]),
    "rd4_b_gate_dir": ("gate", "reddit_llm_dir", ["llm_reddit_v4"]),
    "rd4_c_radd_dir": ("radd", "reddit_llm_dir", ["llm_reddit_v4"]),
    "llm6_d0_lgb_aggr": ("lgb", None, ["llm_news_v6", "llm_reddit_v4"]),  # 기준: lgb_v1_aggr
    "llm6_d_lgb_aggr_news": ("lgb", NEWS_F, ["llm_news_v6"]),
    "rd4_d_lgb_aggr_reddit": ("lgb", REDDIT_F, ["llm_reddit_v4"]),
    "llm6rd4_d_lgb_aggr_both": ("lgb", NEWS_F + REDDIT_F, ["llm_news_v6", "llm_reddit_v4"]),
}

# extra: 결과 보기 전에 정함
BOOT_N, BOOT_SEED = 2000, 0                 # 날짜 단위 bootstrap
PLACEBO_N, PLACEBO_SEED = 30, 0             # 가짜 피처: 관측 행끼리 값 섞기 (결측 위치 그대로)
PLACEBO_EXPS = ["llm6_b_gate_dir", "llm6_c_radd_dir", "rd4_b_gate_dir", "rd4_c_radd_dir"]
ADOPT = {"fold4_min": 0.0, "up_folds_min": 3, "mean_min": 0.005}   # 채택 기준 (팀 합의, 그대로)


# ================================================================ 데이터

def load_source(src):
    s = SOURCES[src]
    n = pd.read_csv(s["dir"] / "daily_features.csv", encoding="utf-8-sig", parse_dates=["target"])
    assert n.target.max() < pd.Timestamp(HOLDOUT), "홀드아웃 날짜가 들어 있음"
    assert not n.duplicated(["symbol", "target"]).any()
    x = n[["symbol", "target", *s["rename"]]].rename(columns=s["rename"])
    x[f"{src}_llm_dir"] = x[s["pos"]] - x[s["neg"]]       # 하나라도 결측이면 결측
    x[f"{src}_llm_surp"] = x[s["pos"]] + x[s["neg"]]
    return n, x


def build_master(ds):
    if E._master is not None and "news_llm_dir" in E._master:
        return E._master
    m = E.master(ds.split(HOLDOUT)[0])
    m2 = m
    for src in SOURCES:
        _, x = load_source(src)
        m2 = m2.merge(x, on=["symbol", "target"], how="left", validate="one_to_one")
    assert len(m2) == len(m)
    E._master = m2
    return m2


def llm_table():
    """symbol + target 한 줄. LightGBM 경로(work/a/lgb.py 의 텍스트 표)에 붙일 LLM 열."""
    a = load_source("news")[1]
    b = load_source("reddit")[1]
    return a.merge(b, on=["symbol", "target"], how="outer", validate="one_to_one")


# ================================================================ 실험 방식

def _std_obs(x):
    """관측 행만으로 평균·표준편차. 결측은 결측 그대로 둠."""
    x = np.asarray(x, float)
    return np.nanmean(x), (np.nanstd(x) or 1.0)


def _rule_v(s1, x):
    vv = np.asarray(s1(x), float)
    vv[np.isnan(x.gap_z.to_numpy(float))] = np.nan      # 갭이 없으면 보합 (B 규칙과 같게)
    return vv


def fit_rsize(t, f):
    """a. B 규칙 그대로, 급등락 경계만 b·exp(−δ·x̃). x̃ 결측이면 경계 b."""
    s1, _, i1 = fit_rule(t, [])
    ms = _std_obs(t[f])
    y = t.label.to_numpy(int)
    v = _rule_v(s1, t)

    def xt(x):
        return np.clip((np.asarray(x[f], float) - ms[0]) / ms[1], -5, 5)

    def lab(v, xf, a, b, d):
        o = E.R.cut(v, a, np.inf)
        bb = np.where(np.isnan(xf), b, b * np.exp(-d * np.nan_to_num(xf)))
        o[v >= bb] = 4
        o[v <= -bb] = 0
        o[np.isnan(v)] = 2
        return o
    xf = xt(t)
    q = np.quantile(np.abs(v[~np.isnan(v)]), E.QS)
    sc, a, b, d = max(((score(y, lab(v, xf, a, b, d))["score"], a, b, d)
                       for i, a in enumerate(q) for b in q[i + 1:] for d in DELTAS), key=lambda r: r[0])
    return (lambda x: lab(_rule_v(s1, x), xt(x), a, b, d)), {
        "lam": i1["lam"], "delta": d, "a": a, "b": b, "train_score": sc,
        "x_mean": ms[0], "x_sd": ms[1], "train_obs_rows": int(np.isfinite(xf).sum())}


def _gate(o, v, dr, mode, tau):
    o = o.copy()
    if mode == "off":
        return o
    m = np.isfinite(dr) & (np.abs(dr) > tau) & (dr != 0) & (np.sign(dr) != np.sign(v)) & np.isin(o, [0, 4])
    if mode == "one_step":
        o[m & (o == 0)] = 1
        o[m & (o == 4)] = 3
    else:
        o[m] = 2
    return o


def fit_gate(t, f):
    """b. B 규칙 등급을 낸 뒤, 방향이 반대인 급등락만 낮춤. 경계는 B 규칙 그대로."""
    s1, (a, b), i1 = fit_rule(t, [])
    y = t.label.to_numpy(int)
    v = _rule_v(s1, t)
    o = E.R.cut(v, a, b)
    dr = t[f].to_numpy(float)
    cands = [("off", 0.0)] + [(m, tau) for m in GATE_MODES[1:] for tau in GATE_TAUS]
    sc, mode, tau = max(((score(y, _gate(o, v, dr, m, tau))["score"], m, tau) for m, tau in cands),
                        key=lambda r: r[0])

    def labels(x):
        vv = _rule_v(s1, x)
        return _gate(E.R.cut(vv, a, b), vv, x[f].to_numpy(float), mode, tau)
    return labels, {"lam": i1["lam"], "a": a, "b": b, "gate_mode": mode, "gate_tau": tau, "train_score": sc,
                    "train_changed_rows": int((_gate(o, v, dr, mode, tau) != o).sum())}


def fit_radd(t, f):
    """c. s + β·sd(s)·dir. dir 결측이면 더하는 항 없음 (규칙 점수 그대로). 경계는 β 마다 train 에서."""
    s1, _, i1 = fit_rule(t, [])
    y = t.label.to_numpy(int)
    v = _rule_v(s1, t)
    sd = np.nanstd(v)

    def add(vv, dr, bt):
        return vv + np.where(np.isfinite(dr), bt * sd * np.nan_to_num(dr), 0.0)
    dr = t[f].to_numpy(float)
    sc, a, b, bt = max(((*E.fit_cut(add(v, dr, bt), y), bt) for bt in BETAS), key=lambda r: r[0])
    return (lambda x: E.R.cut(add(_rule_v(s1, x), x[f].to_numpy(float), bt), a, b)), {
        "lam": i1["lam"], "beta": bt, "a": a, "b": b, "train_score": sc}


def fit_base(t, f=None):
    s1, (a, b), i1 = fit_rule(t, [])
    return (lambda x: E.R.cut(_rule_v(s1, x), a, b)), {"lam": i1["lam"], "a": a, "b": b,
                                                       "train_score": i1["train_score"]}


FITS = {"rule": fit_base, "rsize": fit_rsize, "gate": fit_gate, "radd": fit_radd}
_lgb_ready = False


def make_fit(kind, f):
    global _lgb_ready
    if kind == "lgb":
        if f and not _lgb_ready:      # C 텍스트 표(symbol+target)에 LLM 열만 붙임. lgb.py 는 그대로
            c = pd.read_parquet(L.TEXT_TABLE).drop(columns=["date", "cutoff"])
            L._text = c.merge(llm_table(), on=["symbol", "target"], how="left", validate="one_to_one")
            _lgb_ready = True
        return L.make_fit("aggr", B_FEATURES_V1 + list(f or []))

    def fit(train_days, exclude):
        labels, info = FITS[kind](E.train_rows(train_days, exclude), f)
        return E.TableModel(labels, info)
    fit.__qualname__ = f"fit_llm_{kind}"
    return fit


# ================================================================ 1. check

def _pct(a, b):
    return f"{a}/{b} = {100 * a / b:.1f}%" if b else "0/0"


def _signal(mm, src):
    s = SOURCES[src]
    out = {}
    for nm, sel in [("pos>0", mm[s["pos"]] > 0), ("neg>0", mm[s["neg"]] > 0),
                    ("neither", (mm[s["pos"]] == 0) & (mm[s["neg"]] == 0))]:
        x = mm[sel]
        out[nm] = {"n": len(x), "up(ret>0)": _pct(int((x.ret_pct > 0).sum()), len(x)),
                   "big(label0,4)": _pct(int(x.label.isin([0, 4]).sum()), len(x)),
                   "mean_ret_pct": round(float(x.ret_pct.mean()), 3)}
    d = mm[(mm[f"{src}_llm_dir"] != 0) & mm[f"{src}_llm_dir"].notna() & mm.gap_z.notna() & (mm.gap_z != 0)]
    agree = np.sign(d[f"{src}_llm_dir"]) == np.sign(d.gap_z)
    dis = d[~agree]
    out["gap_cmp"] = {"n_dir_nonzero_with_gap": len(d), "agree": _pct(int(agree.sum()), len(d)),
                      "disagree_price_went_llm_way": _pct(int((np.sign(dis.ret_pct) == np.sign(dis[f"{src}_llm_dir"])).sum()), len(dis))}
    return out


def check(ds):
    m = build_master(ds)
    dev = ds.split(HOLDOUT)[0]
    cal = pd.DataFrame({"date": [d.date for d in dev], "target": [d.target for d in dev]})
    r = {}
    for src in SOURCES:
        n, x = load_source(src)
        q = {"rows": len(n), "symbols": n.symbol.nunique(), "targets": n.target.nunique(),
             "period_target": [str(n.target.min().date()), str(n.target.max().date())],
             "holdout_rows": int((n.target >= pd.Timestamp(HOLDOUT)).sum()),
             "llm_status": n.llm_status.value_counts().to_dict(),
             "targets_not_in_our_calendar": int((~n.target.isin(cal.target)).sum())}
        if "date" in n:      # 뉴스만 기준일 열이 있음
            j = n[["date", "target"]].drop_duplicates().assign(date=lambda d: pd.to_datetime(d.date, format="ISO8601"))
            j = j.merge(cal, on="date", how="left", suffixes=("", "_ours"))
            q["calendar_mismatch"] = int((j.target != j.target_ours).sum())
        c = n[n.llm_status == "complete"]
        q["complete_mean"] = {v: round(float(c[k].mean()), 4) for k, v in SOURCES[src]["rename"].items()}
        q["notna"] = {v: int(m[v].notna().sum()) for v in [*SOURCES[src]["rename"].values(), f"{src}_llm_dir"]}
        mm = m[m[f"{src}_llm_dir"].notna()]
        q["signal_train_eda"] = _signal(mm[mm.target <= pd.Timestamp("2026-02-12")], src)
        q["signal_fold4_period"] = _signal(mm[mm.target > pd.Timestamp("2026-03-04")], src)
        cov = {}
        for f in load_folds(ds):
            v = m[m.date.isin({dd.date for dd in f.val})]
            dr = v[f"{src}_llm_dir"]
            cov[f.name] = {"val_rows": len(v), "rows_with_llm": int(dr.notna().sum()),
                           "rows_dir_nonzero": int(((dr != 0) & dr.notna()).sum()),
                           "big_rows": int(v.label.isin([0, 4]).sum()),
                           "big_rows_with_dir": int((v.label.isin([0, 4]) & (dr != 0) & dr.notna()).sum())}
        q["val_coverage"] = cov
        r[src] = q
    # 누수: 기사·글 시각
    va = pd.read_csv(SOURCES["news"]["dir"] / "valid_articles.csv", encoding="utf-8-sig",
                     parse_dates=["since", "cutoff", "known_at"])
    r["news"]["leak"] = {"articles": len(va), "known_after_cutoff": int((va.known_at > va.cutoff).sum()),
                         "known_before_since": int((va.known_at < va.since).sum()),
                         "max_known_at": str(va.known_at.max())}
    vp = pd.read_csv(SOURCES["reddit"]["dir"] / "valid_posts.csv", encoding="utf-8-sig",
                     parse_dates=["created_et", "since", "cutoff"])
    r["reddit"]["leak"] = {"posts": len(vp), "created_after_cutoff": int((vp.created_et >= vp.cutoff).sum()),
                           "created_before_since": int((vp.created_et < vp.since).sum()),
                           "max_created_et": str(vp.created_et.max()),
                           "subreddits": vp.subreddit.value_counts().to_dict(),
                           "uses_score_cols": [c for c in vp.columns if c in ("score", "n_comments")]}
    # 첫 판과 dev 판의 겹치는 기간이 같은가
    o = pd.read_csv(CACHE / "llm_news_v6" / "daily_features.csv", encoding="utf-8-sig")
    nn = pd.read_csv(SOURCES["news"]["dir"] / "daily_features.csv", encoding="utf-8-sig")
    nn = nn[nn.target <= o.target.max()].sort_values(["symbol", "target"]).reset_index(drop=True)[o.columns]
    oo = o.sort_values(["symbol", "target"]).reset_index(drop=True)
    r["news"]["first_vs_dev_same"] = bool(nn.shape == oo.shape and
                                          (nn.fillna(-9).astype(str).values == oo.fillna(-9).astype(str).values).all())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check.json").write_text(json.dumps(r, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(r, indent=2, ensure_ascii=False, default=str))


# ================================================================ 2. run

def _summ(res):
    t = res.table
    return (t[t.group == "all"].set_index("fold").score, t[t.group == "unseen"].set_index("fold").score)


def run(ds, names):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.json"
    allres = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for nm in names:
        kind, f, tags = EXPS[nm]
        t0 = time.time()
        models = {}

        def fit(train_days, exclude, _f=make_fit(kind, f)):
            mdl = _f(train_days, exclude)
            models[train_days[-1].date] = getattr(mdl, "info", {})
            return mdl
        fit.__qualname__ = f"fit_{kind}"
        res = run_cv(fit, ds, name=nm, method=METHOD, tags=tags + [kind], verbose=False,
                     config={"model": kind, "features": f, "grid": {"deltas": DELTAS, "gate_modes": GATE_MODES,
                             "gate_taus": GATE_TAUS, "betas": BETAS}, "llm_model": "gemini-3.1-flash-lite",
                             "llm_data": "chaemin v6 news / v4 reddit dev handoff final",
                             "main_metric": "fold4/all/score"})
        a, u = _summ(res)
        res.preds.to_parquet(OUT / f"preds_{nm}.parquet", index=False)
        info = {str(k.date()): {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                                for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in models.items()}
        allres[nm] = {"all": a.to_dict(), "unseen": u.to_dict(), "mean": float(a.mean()),
                      "params": info, "seconds": round(time.time() - t0)}
        print(f"{nm:28s} " + " ".join(f"{v:.4f}" for v in a) + f" | mean {a.mean():.4f} | unseen "
              + " ".join(f"{v:.4f}" for v in u) + f" ({time.time() - t0:.0f}s)", flush=True)
        path.write_text(json.dumps(allres, indent=2, default=str), encoding="utf-8")


# ================================================================ 3. extra

def _boot_diff(pa, pb, rng):
    """같은 val 날짜를 다시 뽑아 (실험 − 기준) score 차이 분포."""
    j = pa.merge(pb[["date", "symbol", "label_pred"]], on=["date", "symbol"], suffixes=("_a", "_b"))
    g = [x for _, x in j.groupby("date")]
    out = []
    for _ in range(BOOT_N):
        s = pd.concat([g[i] for i in rng.integers(0, len(g), len(g))])
        out.append(score(s.label, s.label_pred_a)["score"] - score(s.label, s.label_pred_b)["score"])
    return np.array(out)


def ref_of(nm):
    return "llm6_d0_lgb_aggr" if EXPS[nm][0] == "lgb" else "llm6_base"


def extra(ds):
    m = E._master
    runs = json.loads((OUT / "runs.json").read_text(encoding="utf-8"))
    P = {nm: pd.read_parquet(OUT / f"preds_{nm}.parquet") for nm in runs}
    r = {"adopt": {}, "boot": {}, "placebo": {}, "wrongway": {}}
    folds = [f.name for f in load_folds(ds)]
    for nm in runs:
        ref = ref_of(nm)
        if nm == ref:
            continue
        d = {fo: runs[nm]["all"][fo] - runs[ref]["all"][fo] for fo in folds}
        r["adopt"][nm] = {"ref": ref, "diff": d, "mean_diff": float(np.mean(list(d.values()))),
                          "up_folds": int(sum(v > 0 for v in d.values())),
                          "pass": bool(d["fold4"] >= ADOPT["fold4_min"] and sum(v > 0 for v in d.values()) >= ADOPT["up_folds_min"]
                                       and np.mean(list(d.values())) >= ADOPT["mean_min"])}
        rng = np.random.default_rng(BOOT_SEED)
        r["boot"][nm] = {}
        for fo in ["fold4", "all4"]:
            pa = P[nm] if fo == "all4" else P[nm][P[nm].fold == fo]
            pb = P[ref] if fo == "all4" else P[ref][P[ref].fold == fo]
            dd = _boot_diff(pa, pb, rng)
            r["boot"][nm][fo] = {"lo": float(np.quantile(dd, .025)), "hi": float(np.quantile(dd, .975)),
                                 "p_le0": float((dd <= 0).mean())}
    # 가짜 피처: 관측 행끼리 값을 섞어 같은 실험을 다시 돌림 (결측 위치는 그대로)
    for nm in PLACEBO_EXPS:
        kind, f, _ = EXPS[nm]
        prng = np.random.default_rng(PLACEBO_SEED)
        f4, mean = [], []
        for _ in range(PLACEBO_N):
            fake = m.copy()
            ok = fake[f].notna().to_numpy()
            fake.loc[ok, f] = prng.permutation(fake.loc[ok, f].to_numpy())
            E._master = fake
            a, _ = _summ(run_cv(make_fit(kind, f), ds, track=False, verbose=False))
            f4.append(float(a["fold4"]))
            mean.append(float(a.mean()))
        E._master = m
        r["placebo"][nm] = {"real_fold4": runs[nm]["all"]["fold4"], "real_mean": runs[nm]["mean"],
                            "fake_fold4": f4, "fake_mean": mean,
                            "fake_fold4_ge_real": int(sum(x >= runs[nm]["all"]["fold4"] for x in f4)),
                            "fake_mean_ge_real": int(sum(x >= runs[nm]["mean"] for x in mean))}
        (OUT / "extra.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    # 규칙이 급등락을 반대로 찍은 행: LLM 방향이 정답 쪽이었나
    b = P["llm6_base"].merge(m[["target", "symbol", "news_llm_dir", "reddit_llm_dir"]].rename(columns={"target": "date"}),
                             on=["date", "symbol"], how="left")
    ww = b[((b.label == 0) & (b.label_pred >= 3)) | ((b.label == 4) & (b.label_pred <= 1))]
    for fo, x in list(ww.groupby("fold")) + [("all4", ww)]:
        r["wrongway"][fo] = {"rows": len(x)}
        for src in SOURCES:
            dd = x[(x[f"{src}_llm_dir"] != 0) & x[f"{src}_llm_dir"].notna()]
            right = np.sign(dd[f"{src}_llm_dir"]) == np.where(dd.label == 4, 1, -1)
            r["wrongway"][fo][src] = {"with_llm": int(x[f"{src}_llm_dir"].notna().sum()),
                                      "with_dir": len(dd), "llm_right_way": int(right.sum())}
    (OUT / "extra.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if k != "placebo"}, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
                      for k, v in r["placebo"].items()}, indent=2))


# ================================================================ 4. table (experiment_table.md 형식)

WHAT = {   # 바꾼 것 (experiment_table.md 의 '바꾼 것' 열)
    "llm6_base": "기준 재현: v2_rule_base 와 같음 (dev LLM 표를 붙인 master 로)",
    "llm6_a_rsize_relevant": "급등락 경계 b·exp(−δ·x̃(뉴스 관련 비율)), 결측이면 b",
    "llm6_a_rsize_surp": "급등락 경계 ← 뉴스 상회+하회 비율",
    "llm6_b_gate_dir": "뉴스 방향(상회−하회)이 규칙과 반대인 0·4 를 낮춤 {off/한 칸/보합}×τ",
    "llm6_c_radd_dir": "s + β·sd·뉴스 방향(상회−하회)",
    "rd4_a_rsize_direct": "급등락 경계 ← Reddit 직접 관련 비율",
    "rd4_a_rsize_surp": "급등락 경계 ← Reddit 상승+하락 의견 비율",
    "rd4_b_gate_dir": "Reddit 방향(상승−하락 의견)이 규칙과 반대인 0·4 를 낮춤",
    "rd4_c_radd_dir": "s + β·sd·Reddit 방향",
    "llm6_d0_lgb_aggr": "기준 재현: lgb_v1_aggr 와 같음",
    "llm6_d_lgb_aggr_news": "피처: + 뉴스 LLM 4개 (관련·상회·하회·방향), 결측 그대로",
    "rd4_d_lgb_aggr_reddit": "피처: + Reddit LLM 4개 (직접·상승·하락·방향)",
    "llm6rd4_d_lgb_aggr_both": "피처: + 뉴스 4 + Reddit 4",
}
RULE_F4 = 0.5154      # rule_gapz_ext fold4 (experiment_table.md 기준)


def table(ds):
    runs = json.loads((OUT / "runs.json").read_text(encoding="utf-8"))
    ad = json.loads((OUT / "extra.json").read_text(encoding="utf-8"))["adopt"] if (OUT / "extra.json").exists() else {}
    rows = []
    for nm, v in runs.items():
        p = pd.read_parquet(OUT / f"preds_{nm}.parquet")
        # 월 = 대상일 연-월, 모든 폴드 예측을 묶음 (lab/tracking.py 의 monthly 와 같은 정의. 2026-03 에는 fold3 끝 3일도 들어감)
        ym = pd.to_datetime(p.date).dt.strftime("%Y-%m")
        mon = [score(x.label, x.label_pred)["score"] for _, x in p[ym.isin(["2026-03", "2026-04", "2026-05"])].groupby(ym)]
        a = v["all"]
        ps = list(v["params"].values())
        keys = [k for k in ["lam", "delta", "gate_mode", "gate_tau", "beta", "k", "rounds"] if any(k in q for q in ps)]
        par = "; ".join(f"{k} " + "/".join(str(round(q[k], 3)) if isinstance(q.get(k), float) else str(q.get(k)) for q in ps)
                        for k in keys)
        ref = ref_of(nm) if nm not in ("llm6_base", "llm6_d0_lgb_aggr") else ("v2_rule_base" if nm == "llm6_base" else "lgb_v1_aggr")
        up = ad.get(nm, {}).get("up_folds", "")
        rows.append(f"| {nm} | {ref} | {WHAT[nm]} | " + " | ".join(f"{a[f]:.4f}" for f in ["fold1", "fold2", "fold3"])
                    + f" | {a['fold4']:.4f} | {v['mean']:.4f} | {v['unseen']['fold4']:.4f} | "
                    + " | ".join(f"{m:.4f}" for m in mon) + f" | {a['fold4'] - RULE_F4:+.4f} | {up} | {par} |")
    head = ("| run | 비교 대상 | 바꾼 것 | f1 | f2 | f3 | **f4** | 평균 | unseen | 3월 | 4월 | 5월 | vs 규칙 f4 | 상승 폴드 | 학습된 파라미터 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    txt = head + "\n" + "\n".join(rows)
    (OUT / "table.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "table":
        table(None)
        sys.exit()
    ds = Dataset()
    build_master(ds)
    if cmd == "check":
        check(ds)
    elif cmd == "run":
        run(ds, sys.argv[2:] or list(EXPS))
    elif cmd == "extra":
        extra(ds)
