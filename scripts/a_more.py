"""추가 실험 (2026-10-07 밤): 규칙의 흔들림 줄이기 + 시장 상태로 급등락 경계 옮기기.

    uv run python scripts/a_more.py run [이름 ...]     # W&B tags ["more_v1", 방식]
    uv run python scripts/a_more.py extra              # bootstrap + 가짜 피처 + 채택 판정

근거 (EDA 기간 = 대상일 ≤ 2026-02-12 만 보고 정함. val 은 안 봄. worklog 2026-10-07 §11)
    - 학습 기간만 바꾼 v3_recent250 / v3_winvote 가 fold4 에서 가장 좋았음 → 경계 a·b 가 train 표본에 따라
      흔들리는 문제(분산)를 줄이면 오를 수 있음 → bag(날짜 bootstrap 30개 규칙의 등급 중간값), innerwin
      (train 마지막 60일로 기간을 고름. val 을 안 보고 고르는 recent 계열)
    - VIX(외부, CBOE) 상위 1/3 구간의 급등락 비율 17.4% vs 하위 10.8% → 크기 신호. 기준일 종가(16:15 ET 확정)만 씀
    - 급등락이 날짜에 몰림 (하루 급등락 비율 99% 분위 59%) → 그날 전 종목 |gap_z| 중앙값 (cutoff 에 앎)
    - 애널리스트 장 마감 후 의견(an_now_net): 갭과 엇갈리면 주가가 애널리스트 쪽으로 간 비율 43% → 기대 낮음, 대조로 하나만
외부 데이터: cache/external/VIX_History.csv (cdn.cboe.com VIX_History.csv). 기준일 < 2026-06-01 만 읽음.
격자는 a_llm_news.py 와 같은 DELTAS / BETAS (새로 만들지 않음).
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
from work.a.models2 import fit_rule  # noqa: E402

OUT = CACHE / "more_v1"
VIX = CACHE / "external" / "VIX_History.csv"
TAGS = ["more_v1"]

# ================================================================ 고정 (실행 전에 정함)
BAG_N, BAG_SEED = 30, 0               # 날짜 bootstrap 규칙 수
INNER_DAYS = 60                       # innerwin: train 끝 60 기준일로 기간 고르기
WINDOWS = [10 ** 6, 250, 120]         # 전체 / 최근 250 / 최근 120 기준일 (v3 와 같은 후보)
PLACEBO_N, PLACEBO_SEED = 30, 0

EXPS = {   # 이름: (방식, 피처)
    "more_bag30": ("bag", None),
    "more_innerwin": ("innerwin", None),
    "more_rsize_vix": ("rsize", "vix"),               # VIX 종가 (기준일)
    "more_rsize_vixchg5": ("rsize", "vix_chg5"),      # log(VIX / 5거래일 전 VIX)
    "more_rsize_dayabsgap": ("rsize", "day_absgap"),  # 그날 전 종목 |gap_z| 중앙값
    "more_radd_analyst": ("radd", "an_dir"),          # 장 마감 후 애널리스트 순의견, ±3 에서 자르고 ÷3
}
# 2차 (1차 결과를 본 뒤, EDA 기간 근거로 추가. worklog §11.4)
#   정답 등급은 절대 수익률(±0.6%, ±2.5%)로 나뉘는데 B 규칙은 gap/vol20 로만 자름.
#   EDA: 0.5<|gap_z|<1 행의 급등락 확률이 저변동 7% / 고변동 41%. 장중 잔차 sd ≈ 0.005 + 0.7·vol20, 갭 되돌림 기울기 −0.14
#   구조: ret = (1−ρ)·g·exp(−λ·x̃(ext_range_z)) + ε, ε ~ N(0, (c0 + c1·vol20)²) → 등급 확률 → 기대 등급 − 2 → |s| 경계 (fit_cut)
RHOS = [0.0, 0.15, 0.3]
C0S = [0.0, 0.003, 0.006]
C1S = [0.6, 0.8, 1.0]
STRUCT_LAMS = [0.0, 0.1, 0.2, 0.3, 0.4]          # models2.LAMS 와 같음
EXPS.update({
    "more_struct": ("struct", None),              # λ = 0 고정
    "more_struct_ext": ("struct_ext", None),      # λ 도 train 에서
})
PLACEBO_EXPS = ["more_rsize_vix", "more_rsize_vixchg5", "more_rsize_dayabsgap", "more_radd_analyst"]
DAY_LEVEL = {"vix", "vix_chg5", "day_absgap"}       # 가짜 피처는 날짜 단위로 섞음


# ================================================================ 데이터

def add_cols(m):
    v = pd.read_csv(VIX, parse_dates=["DATE"], date_format="%m/%d/%Y")
    v = v[v.DATE < pd.Timestamp(HOLDOUT)].sort_values("DATE")      # 홀드아웃 행은 읽자마자 버림
    v["vix_chg5"] = np.log(v.CLOSE / v.CLOSE.shift(5))
    v = v.rename(columns={"DATE": "date", "CLOSE": "vix"})[["date", "vix", "vix_chg5"]]
    # 기준일 VIX 종가는 기준일 16:15 ET 에 확정 → 대상일 09:30 cutoff 전에 앎
    m = m.drop(columns=[c for c in ("vix", "vix_chg5", "day_absgap", "an_dir") if c in m])
    m = m.merge(v, on="date", how="left", validate="many_to_one")
    m["day_absgap"] = m.gap_z.abs().groupby(m.date).transform("median")
    m["an_dir"] = m.an_now_net.clip(-3, 3) / 3
    return m


def build(ds):
    m = add_cols(N.build_master(ds))
    assert m.vix.notna().all(), "VIX 가 빈 기준일이 있음"
    E._master = m
    return m


# ================================================================ 방식

def _cut_labels(s1, a, b):
    return lambda x: E.R.cut(N._rule_v(s1, x), a, b)


def fit_bag(t):
    """날짜를 중복 허용으로 다시 뽑아 B 규칙 30개 → 행마다 등급 중간값."""
    rng = np.random.default_rng(BAG_SEED)
    by = {d: g for d, g in t.groupby("date")}
    dates = np.array(sorted(by))
    fits = []
    for _ in range(BAG_N):
        tb = pd.concat([by[d] for d in rng.choice(dates, len(dates))], ignore_index=True)
        s1, (a, b), i1 = fit_rule(tb, [])
        fits.append((s1, a, b, i1["lam"]))

    def labels(x):
        return np.median(np.vstack([E.R.cut(N._rule_v(s, x), a, b) for s, a, b, _ in fits]), axis=0).astype(int)
    return labels, {"lams": [f[3] for f in fits], "b_med": float(np.median([f[2] for f in fits]))}


def fit_innerwin(t):
    """train 끝 60일을 떼어 각 기간으로 학습 → 그 60일 score 최대 기간을 고르고, 전체 train 의 같은 기간으로 다시 학습."""
    ud = np.sort(t.date.unique())
    inner = set(ud[-INNER_DAYS:])
    pre = t[~t.date.isin(inner)]
    pud = np.sort(pre.date.unique())
    val = t[t.date.isin(inner)]
    sc = {}
    for n in WINDOWS:
        s1, (a, b), _ = fit_rule(pre[pre.date >= pud[-min(n, len(pud))]], [])
        sc[n] = score(val.label, E.R.cut(N._rule_v(s1, val), a, b))["score"]
    n = max(sc, key=sc.get)
    s1, (a, b), i1 = fit_rule(t[t.date >= ud[-min(n, len(ud))]], [])
    return _cut_labels(s1, a, b), {"window": int(min(n, len(ud))), "inner_scores": {str(k): v for k, v in sc.items()},
                                   "lam": i1["lam"], "a": a, "b": b}


EDGES = np.array([-0.025, -0.006, 0.006, 0.025])   # 등급 경계 (−2.5%, −0.6%, +0.6%, +2.5%)


def _struct_s(x, rho, c0, c1, lam, me):
    from scipy.stats import norm
    g = x.gap.to_numpy(float) * np.exp(-lam * E.zapply(x.ext_range_z, me))
    mu = (1 - rho) * g
    sd = c0 + c1 * x.vol20.to_numpy(float)
    cdf = norm.cdf((EDGES[None, :] - mu[:, None]) / sd[:, None])          # P(ret < 경계)
    p = np.diff(np.hstack([np.zeros((len(g), 1)), cdf, np.ones((len(g), 1))]), axis=1)
    s = p @ E.LV - 2
    s[np.isnan(g) | np.isnan(sd) | ~np.isfinite(x.gap_z.to_numpy(float))] = np.nan
    return s


def fit_struct(t, lams):
    me = E.zfit(t.ext_range_z)
    y = t.label.to_numpy(int)
    best = max(((*E.fit_cut(_struct_s(t, r, c0, c1, lam, me), y), r, c0, c1, lam)
                for r in RHOS for c0 in C0S for c1 in C1S for lam in lams), key=lambda z: z[0])
    sc, a, b, r, c0, c1, lam = best
    return (lambda x: E.R.cut(_struct_s(x, r, c0, c1, lam, me), a, b)), {
        "rho": r, "c0": c0, "c1": c1, "lam": lam, "a": a, "b": b, "train_score": sc}


def make_fit(kind, f):
    def fit(train_days, exclude):
        t = E.train_rows(train_days, exclude)
        if kind in ("struct", "struct_ext"):
            labels, info = fit_struct(t, [0.0] if kind == "struct" else STRUCT_LAMS)
        elif kind == "bag":
            labels, info = fit_bag(t)
        elif kind == "innerwin":
            labels, info = fit_innerwin(t)
        else:
            labels, info = N.FITS[kind](t, f)
        return E.TableModel(labels, info)
    fit.__qualname__ = f"fit_more_{kind}"
    return fit


# ================================================================ run / extra

def run(ds, names):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.json"
    allres = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for nm in names:
        kind, f = EXPS[nm]
        t0, infos = time.time(), {}

        def fit(train_days, exclude, _f=make_fit(kind, f)):
            mdl = _f(train_days, exclude)
            infos[str(train_days[-1].date.date())] = mdl.info
            return mdl
        fit.__qualname__ = f"fit_more_{kind}"
        res = run_cv(fit, ds, name=nm, method=N.METHOD, tags=TAGS + [kind], verbose=False,
                     config={"model": kind, "feature": f, "deltas": N.DELTAS, "betas": N.BETAS,
                             "bag_n": BAG_N, "inner_days": INNER_DAYS, "windows": WINDOWS,
                             "external": "CBOE VIX_History.csv (기준일 종가)" if f and "vix" in f else None,
                             "main_metric": "fold4/all/score"})
        a, u = N._summ(res)
        res.preds.to_parquet(OUT / f"preds_{nm}.parquet", index=False)
        allres[nm] = {"all": a.to_dict(), "unseen": u.to_dict(), "mean": float(a.mean()),
                      "params": json.loads(json.dumps(infos, default=float)), "seconds": round(time.time() - t0)}
        print(f"{nm:24s} " + " ".join(f"{v:.4f}" for v in a) + f" | mean {a.mean():.4f} | unseen4 {u['fold4']:.4f}"
              f" ({time.time() - t0:.0f}s)", flush=True)
        path.write_text(json.dumps(allres, indent=2), encoding="utf-8")


def extra(ds):
    m = E._master
    runs = json.loads((OUT / "runs.json").read_text(encoding="utf-8"))
    base = json.loads((N.OUT / "runs.json").read_text(encoding="utf-8"))["llm6_base"]
    pb = pd.read_parquet(N.OUT / "preds_llm6_base.parquet")
    r = {"adopt": {}, "boot": {}, "placebo": {}}
    for nm, v in runs.items():
        d = {fo: v["all"][fo] - base["all"][fo] for fo in base["all"]}
        r["adopt"][nm] = {"diff": d, "mean_diff": float(np.mean(list(d.values()))),
                          "up_folds": int(sum(x > 0 for x in d.values())),
                          "pass": bool(d["fold4"] >= 0 and sum(x > 0 for x in d.values()) >= 3
                                       and np.mean(list(d.values())) >= 0.005)}
        pa = pd.read_parquet(OUT / f"preds_{nm}.parquet")
        rng = np.random.default_rng(N.BOOT_SEED)
        r["boot"][nm] = {}
        for fo in ["fold4", "all4"]:
            a_ = pa if fo == "all4" else pa[pa.fold == fo]
            b_ = pb if fo == "all4" else pb[pb.fold == fo]
            dd = N._boot_diff(a_, b_, rng)
            r["boot"][nm][fo] = {"lo": float(np.quantile(dd, .025)), "hi": float(np.quantile(dd, .975)),
                                 "p_le0": float((dd <= 0).mean())}
    for nm in PLACEBO_EXPS:
        kind, f = EXPS[nm]
        prng = np.random.default_rng(PLACEBO_SEED)
        f4, mean = [], []
        for _ in range(PLACEBO_N):
            fake = m.copy()
            if f in DAY_LEVEL:      # 날짜 단위 값은 날짜끼리 섞음 (같은 날 종목은 같은 값 유지)
                dv = fake.groupby("date")[f].first()
                fake[f] = fake.date.map(pd.Series(prng.permutation(dv.to_numpy()), index=dv.index))
            else:
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
        (OUT / "extra.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    (OUT / "extra.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if k != "placebo"}, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in r["placebo"].items()}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    ds = Dataset()
    build(ds)
    if cmd == "run":
        run(ds, sys.argv[2:] or list(EXPS))
    elif cmd == "extra":
        extra(ds)
