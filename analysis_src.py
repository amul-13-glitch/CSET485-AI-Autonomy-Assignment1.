# %% [markdown]
# # CSET485 – AI and Society · Assignment #1
# **Stress-testing a Credit Approval Prediction Model (German Credit Data)**
#
# Topic lens: *Does AI reduce human autonomy in decision making?*
#
# Student: Amul Sharma · Enrollment S24BCAU0013 · Last digit 3 → German Credit · Last three digits 013 → `random_state=13`
#
# Run top-to-bottom (`Kernel → Restart & Run All`). Every number in the PDF report is written to
# `outputs/results.json` by this notebook, and the tables/figures are written to `outputs/`.

# %%
import json, os, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

STUDENT, ENROLL, SEED = "Amul Sharma", "S24BCAU0013", 13   # SEED = last three digits of the roll number
DATA = "data/german.data"                                    # original UCI symbolic file (committed to the repo)
os.makedirs("outputs", exist_ok=True); os.makedirs("data", exist_ok=True)
if not os.path.exists(DATA):   # fetch the original UCI symbolic file (public mirror fallback)
    import urllib.request
    for url in ["https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
                "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"]:
        try:
            txt = urllib.request.urlopen(url, timeout=30).read().decode().replace(",", " ")
            open(DATA, "w").write(txt); break
        except Exception as e: print("download failed:", url, e)
R = {}                                                       # every reported number is stored here

# %% [markdown]
# ## 0. Load data and preprocessing
# * Source: UCI Statlog (German Credit Data), `german.data` (symbolic version, 1,000 rows, 20 inputs + class).
# * Class 1 = Good → approve (1); class 2 = Bad → reject (0). This is the assignment's *operational* mapping,
#   not a record of real lending decisions.

# %%
cols = ["checking_status","duration","credit_history","purpose","credit_amount","savings","employment",
        "installment_rate","personal_status","other_debtors","residence","property","age","other_plans",
        "housing","existing_credits","job","people_liable","telephone","foreign_worker","target"]
df = pd.read_csv(DATA, sep=" ", header=None, names=cols)
df["y"] = (df["target"] == 1).astype(int)
NUM = ["duration","credit_amount","installment_rate","residence","age","existing_credits","people_liable"]
CAT = [c for c in cols if c not in NUM + ["target"]]
R["n_rows"], R["n_good"], R["n_bad"] = len(df), int(df.y.sum()), int((1-df.y).sum())
R["missing_values"] = int(df.isna().sum().sum())
print(df.shape, "missing:", R["missing_values"], "| good/bad:", R["n_good"], R["n_bad"])

# Outliers: 1.5×IQR flags (inspected, nothing deleted)
iqr = {}
for c in NUM:
    q1, q3 = df[c].quantile([.25,.75]); i = q3-q1
    iqr[c] = int(((df[c] < q1-1.5*i) | (df[c] > q3+1.5*i)).sum())
R["iqr_flags"] = iqr; print(iqr)

# Split: 70/30, no stratification, random_state = last three digits of roll number
tr_df, te_df = train_test_split(df, test_size=0.30, random_state=SEED)
R["train_n"], R["test_n"] = len(tr_df), len(te_df)
R["train_good"], R["train_bad"] = int(tr_df.y.sum()), int((1-tr_df.y).sum())
R["test_good"], R["test_bad"] = int(te_df.y.sum()), int((1-te_df.y).sum())

# One-hot encode all 13 categorical attributes; reference level = first sorted TRAINING category
def design(train, test):
    levels = {c: sorted(train[c].unique()) for c in CAT}
    def enc(d):
        out = d[NUM].astype(float).copy()
        for c in CAT:
            for lv in levels[c][1:]:
                out[f"{c}={lv}"] = (d[c] == lv).astype(float)
        return out
    Xtr, Xte = enc(train), enc(test)
    mu, sd = Xtr.mean(), Xtr.std(ddof=0)          # scaling parameters learned on TRAIN only
    keep = sd[sd > 0].index
    Xtr = (Xtr[keep]-mu[keep])/sd[keep]; Xte = (Xte[keep]-mu[keep])/sd[keep]
    return sm.add_constant(Xtr, has_constant="add"), sm.add_constant(Xte, has_constant="add")
Xtr, Xte = design(tr_df, te_df)
ytr, yte = tr_df.y.values, te_df.y.values
R["design_columns_incl_intercept"] = Xtr.shape[1]
print("design:", Xtr.shape, Xte.shape)

# %% [markdown]
# ## Part 1 – Build and compare the models

# %%
def fit_logit(X, y):
    try:
        return sm.Logit(y, X).fit(disp=0, maxiter=200, method="newton")
    except Exception:
        return sm.Logit(y, X).fit(disp=0, maxiter=500, method="bfgs")

ols = sm.OLS(ytr, Xtr).fit()
logit = fit_logit(Xtr, ytr)
pA = ols.predict(Xte).values                       # Model A: OLS "probability"
pB = logit.predict(Xte).values                     # Model B: logistic probability
med = float(np.median(pA))
flagA = (pA >= med).astype(int)                    # high-"approve" by median rule
flagB = (pB >= 0.5).astype(int)
cm = pd.crosstab(flagA, flagB).reindex(index=[0,1], columns=[0,1], fill_value=0)
mismatch = int((flagA != flagB).sum())
R["ols_median"], R["ols_min"], R["ols_max"] = med, float(pA.min()), float(pA.max())
R["ols_outside_01"] = int(((pA < 0) | (pA > 1)).sum())
R["agreement"] = cm.values.tolist()
R["mismatch_n"], R["mismatch_pct"] = mismatch, 100*mismatch/len(yte)
R["logit_approve_n"] = int(flagB.sum()); R["ols_approve_n"] = int(flagA.sum())
R["logit_acc"] = float((flagB == yte).mean()); R["ols_median_acc"] = float((flagA == yte).mean())
R["majority_baseline"] = float(max(yte.mean(), 1-yte.mean()))
R["mismatch_dirs"] = {"olsLow_logitHigh": int(((flagA==0)&(flagB==1)).sum()),
                      "olsHigh_logitLow": int(((flagA==1)&(flagB==0)).sum())}
R["logit_iterations_converged"] = bool(getattr(logit, "mle_retvals", {}).get("converged", True))
print(cm); print({k: R[k] for k in ["ols_median","mismatch_n","mismatch_pct","logit_acc","majority_baseline"]})

# %% [markdown]
# ## 2.1 Simpson's paradox hunt
# Numeric-only logistic regressions (seven standardised numeric predictors) on the full training set, subgroup A and
# subgroup B. We scan several social/demographic splits and every numeric predictor, looking for a sign reversal;
# a Wald z-test for the difference between the two subgroup coefficients is reported for each.

# %%
tr = tr_df.copy()
tr["male"] = tr.personal_status.isin(["A91","A93","A94"])
SPLITS = {"Sex (male vs female)": tr.male,
          "Age (>=35 vs <35)": tr.age >= 35,
          "Telephone registered (yes vs no)": tr.telephone == "A192",
          "Housing (own vs not own)": tr.housing == "A152",
          "Job (skilled+ vs unskilled)": tr.job.isin(["A173","A174"])}

def fit_num(d):
    X = d[NUM].astype(float); X = (X-X.mean())/X.std(ddof=0); X = X.loc[:, X.std() > 0]
    m = sm.Logit(d.y.values, sm.add_constant(X)).fit(disp=0, maxiter=200)
    return m

full_num = fit_num(tr)
scan = []
for sname, mask in SPLITS.items():
    a, b = fit_num(tr[mask]), fit_num(tr[~mask])
    for p in NUM:
        z = (a.params[p]-b.params[p]) / np.hypot(a.bse[p], b.bse[p])
        scan.append(dict(split=sname, predictor=p, full=full_num.params[p], A=a.params[p], B=b.params[p],
                         z_diff=z, p_diff=2*(1-stats.norm.cdf(abs(z))),
                         reversal=bool(np.sign(a.params[p]) != np.sign(b.params[p])),
                         nA=int(mask.sum()), nB=int((~mask).sum())))
scan = pd.DataFrame(scan); scan.to_csv("outputs/simpson_scan.csv", index=False)
# Headline: strongest *significant* sign reversal between subgroups (smallest p for the difference)
cand = scan[scan.reversal & (scan.p_diff < 0.05)].sort_values("p_diff")
head = cand.iloc[0]
R["simpson_headline"] = head.to_dict()
mask = SPLITS[head.split]
a, b = fit_num(tr[mask]), fit_num(tr[~mask]); p = head.predictor
def row(label, m, n):
    ci = m.conf_int().loc[p]
    return dict(group=label, n=int(n), coef=float(m.params[p]), se=float(m.bse[p]), ci_lo=float(ci[0]), ci_hi=float(ci[1]), p=float(m.pvalues[p]))
_base = head.split.split("(")[0].strip()
R["simpson_table"] = [row("All records", full_num, len(tr)), row(f"A – {_base}: " + head.split.split("(")[1].split(" vs")[0], a, mask.sum()),
                      row(f"B – {_base}: " + head.split.split("vs ")[1].rstrip(")"), b, (~mask).sum())]
# Secondary: the demographic variable used in the fairness section (sex)
sx = SPLITS["Sex (male vs female)"]; am, bf = fit_num(tr[sx]), fit_num(tr[~sx])
R["sex_scan"] = scan[scan.split.str.startswith("Sex")].to_dict("records")
R["n_reversals_total"] = int(scan.reversal.sum()); R["n_reversals_significant"] = int(len(cand)); R["scan_size"] = len(scan)
print(pd.DataFrame(R["simpson_table"]).round(4)); print(cand[["split","predictor","full","A","B","z_diff","p_diff"]].round(3))

# Controlled follow-up: does the reversal survive adjusting for the other predictors + an interaction test?
d = tr.copy(); d["grp"] = mask.astype(int).values
Z = d[NUM].astype(float); Z = (Z-Z.mean())/Z.std(ddof=0); Z["grp"] = d.grp.values
Z[p+"_x_grp"] = Z[p]*Z["grp"]
inter = sm.Logit(d.y.values, sm.add_constant(Z)).fit(disp=0, maxiter=200)
R["simpson_interaction"] = dict(coef=float(inter.params[p+"_x_grp"]), p=float(inter.pvalues[p+"_x_grp"]))
# Mix of the grouping variable across approval outcome (the structural reason for the aggregate trend)
R["simpson_group_share_by_amount_tercile"] = (tr.assign(t=pd.qcut(tr.credit_amount, 3, labels=["low","mid","high"]), g=mask.values)
      .groupby("t").g.mean().round(4).to_dict())
R["simpson_group_approval_rate"] = {"A": float(tr[mask].y.mean()), "B": float(tr[~mask].y.mean())}
R["simpson_group_mean_amount"] = {"A": float(tr[mask].credit_amount.mean()), "B": float(tr[~mask].credit_amount.mean())}
print(R["simpson_interaction"], R["simpson_group_share_by_amount_tercile"], R["simpson_group_approval_rate"])

# Figure: coefficient comparison
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for i, r in enumerate(R["simpson_table"][::-1]):
    ax.errorbar(r["coef"], i, xerr=[[r["coef"]-r["ci_lo"]], [r["ci_hi"]-r["coef"]]], fmt="o", capsize=4, color="#1f3b57")
ax.set_yticks(range(3)); ax.set_yticklabels([r["group"] for r in R["simpson_table"][::-1]])
ax.axvline(0, color="grey", lw=1, ls="--"); ax.set_xlabel(f"Logistic coefficient for standardised {p} (95% Wald CI)")
ax.set_title("Sign reversal across subgroups"); plt.tight_layout(); plt.savefig("outputs/fig_simpson.png", dpi=200); plt.close()

# %% [markdown]
# ## 2.2 Omitted-variable bias
# Drop all `checking_status` dummies (a plausible confounder: existing checking-account status) and refit the full logistic model.

# %%
drop_cols = [c for c in Xtr.columns if c.startswith("checking_status=")]
Xtr_r, Xte_r = Xtr.drop(columns=drop_cols), Xte.drop(columns=drop_cols)
logit_r = fit_logit(Xtr_r, ytr)
rows = []
for c in NUM:
    f, r_ = logit.params[c], logit_r.params[c]
    rows.append(dict(predictor=c, full=float(f), reduced=float(r_), abs_change=float(r_-f), pct_change=float(100*(r_-f)/abs(f))))
ovb = pd.DataFrame(rows); ovb.to_csv("outputs/omitted_variable.csv", index=False)
R["ovb"] = rows; R["ovb_dropped_columns"] = drop_cols
R["ovb_max_pct"] = ovb.loc[ovb["pct_change"].abs().idxmax()].to_dict()
R["ovb_max_abs"] = ovb.loc[ovb["abs_change"].abs().idxmax()].to_dict()
pB_r = logit_r.predict(Xte_r).values
R["ovb_acc_full"], R["ovb_acc_reduced"] = float(((pB >= .5) == yte).mean()), float(((pB_r >= .5) == yte).mean())
R["ovb_decision_flips"] = int(((pB >= .5) != (pB_r >= .5)).sum())
print(ovb.round(4)); print(R["ovb_acc_full"], R["ovb_acc_reduced"], R["ovb_decision_flips"])

# %% [markdown]
# ## 2.3 Adversarial subset (confident, wrong predictions)

# %%
te = te_df.copy().reset_index(drop=True)
te["p"] = pB; te["male"] = te.personal_status.isin(["A91","A93","A94"])
fp = (te.p >= .75) & (te.y == 0); fn = (te.p <= .25) & (te.y == 1)
wrong = te[fp | fn].copy(); wrong["kind"] = np.where(wrong.p >= .75, "FP", "FN")
R["hc_total"], R["hc_pct"] = int(len(wrong)), 100*len(wrong)/len(te)
R["hc_fp"], R["hc_fn"] = int(fp.sum()), int(fn.sum())
R["hc_n_confident_all"] = int(((te.p >= .75) | (te.p <= .25)).sum())
sub = wrong if len(wrong) <= 50 else wrong.sample(50, random_state=SEED)
R["hc_subset_n"] = int(len(sub))
def prof(d):
    return {"n": int(len(d)), "male": int(d.male.sum()), "female": int((~d.male).sum()),
            "mean_age": float(d.age.mean()), "median_age": float(d.age.median()),
            "mean_amount": float(d.credit_amount.mean()), "mean_duration": float(d.duration.mean()),
            "age_lt25": int((d.age < 25).sum()), "age_25_39": int(((d.age >= 25) & (d.age < 40)).sum()), "age_ge40": int((d.age >= 40).sum())}
R["hc_profile"], R["test_profile"] = prof(sub), prof(te)
traits = {"Checking account A11 (< 0 DM)": lambda d: d.checking_status == "A11",
          "Checking account A14 (no account)": lambda d: d.checking_status == "A14",
          "Credit amount > 5,000": lambda d: d.credit_amount > 5000,
          "Duration > 24 months": lambda d: d.duration > 24,
          "Credit history A30/A31 (rare categories)": lambda d: d.credit_history.isin(["A30", "A31"]),
          "Single male (A93)": lambda d: d.personal_status == "A93",
          "Savings A61 (< 100 DM)": lambda d: d.savings == "A61",
          "Age < 30": lambda d: d.age < 30,
          "Installment rate = 4": lambda d: d.installment_rate == 4,
          "Housing: own (A152)": lambda d: d.housing == "A152",
          "Purpose: car/radio-TV (A40/A43)": lambda d: d.purpose.isin(["A40","A43"]),
          "Existing credits at other banks (A143 none)": lambda d: d.other_plans == "A143"}
trows = []
for k, f in traits.items():
    s, t = f(sub), f(te)
    trows.append(dict(trait=k, subset_n=int(s.sum()), subset_pct=100*s.mean(), test_n=int(t.sum()), test_pct=100*t.mean(),
                      lift=float(s.mean()/t.mean()) if t.mean() > 0 else np.nan,
                      fisher_p=float(stats.fisher_exact([[s.sum(), len(s)-s.sum()], [t.sum()-s.sum(), len(t)-len(s)-(t.sum()-s.sum())]])[1])))
trait_df = pd.DataFrame(trows).sort_values("fisher_p"); trait_df.to_csv("outputs/adversarial_traits.csv", index=False)
R["hc_traits"] = trait_df.to_dict("records")
# FP / FN breakdown of top traits and the model's learned coefficient for the top trait
R["hc_fp_profile"] = prof(wrong[wrong.kind == "FP"]) if (wrong.kind == "FP").any() else None
R["hc_fn_profile"] = prof(wrong[wrong.kind == "FN"]) if (wrong.kind == "FN").any() else None
R["hc_by_kind_traits"] = {k: {kind: float(f(wrong[wrong.kind == kind]).mean()*100) if (wrong.kind == kind).any() else None for kind in ["FP","FN"]} for k, f in traits.items()}
# Class-specific baselines: false negatives can only come from actual-Good rows, false positives only from actual-Bad rows
def fisher_vs(sub_mask, base_mask):
    a_, n_ = int(sub_mask.sum()), len(sub_mask); b_, m_ = int(base_mask.sum()), len(base_mask)
    return dict(sub_n=a_, sub_total=n_, sub_pct=100*a_/n_, base_n=b_, base_total=m_, base_pct=100*b_/m_,
                fisher_p=float(stats.fisher_exact([[a_, n_-a_], [b_-a_, (m_-n_)-(b_-a_)]])[1]))
rare_hist = lambda d: d.credit_history.isin(["A30", "A31"])
noacct = lambda d: d.checking_status == "A14"
fnw, fpw = wrong[wrong.kind == "FN"], wrong[wrong.kind == "FP"]
R["hc_fn_rare_history"] = fisher_vs(rare_hist(fnw), rare_hist(te[te.y == 1]))
R["hc_fp_no_account"] = fisher_vs(noacct(fpw), noacct(te[te.y == 0]))
R["hc_fn_rare_history_all_test_pct"] = float(100*rare_hist(te).mean())
R["hc_either_pattern_n"] = int((rare_hist(fnw).sum() + noacct(fpw).sum()))
R["hc_train_rare_history"] = dict(n=int(rare_hist(tr_df).sum()), pct=float(100*rare_hist(tr_df).mean()), good_rate=float(tr_df[rare_hist(tr_df)].y.mean()), overall_good_rate=float(tr_df.y.mean()))
R["hc_A30_train"] = dict(n=int((tr_df.credit_history == "A30").sum()), good_rate=float(tr_df[tr_df.credit_history == "A30"].y.mean()))
R["hc_A14_train"] = dict(n=int((tr_df.checking_status == "A14").sum()), good_rate=float(tr_df[tr_df.checking_status == "A14"].y.mean()))
R["hc_A14_test_actual_good_rate"] = float(te[noacct(te)].y.mean()); R["hc_A14_test_n"] = int(noacct(te).sum())
R["logit_hist_coefs"] = {k: [float(logit.params[k]), float(logit.bse[k])] for k in logit.params.index if k.startswith("credit_history") or k == "checking_status=A14"}
wrong.to_csv("outputs/adversarial_subset.csv", index=False)
coefs = logit.params.sort_values(); R["top_logit_coefs"] = {"most_negative": coefs.head(6).to_dict(), "most_positive": coefs.tail(6).to_dict()}
R["logit_selected_coefs"] = {k: float(logit.params[k]) for k in logit.params.index if k in ["duration","credit_amount","installment_rate","age"] or k.startswith("checking_status") or k.startswith("savings")}
# Calibration of confident predictions in the whole test set
R["calib"] = {"p>=0.75": {"n": int((te.p >= .75).sum()), "actual_good_rate": float(te[te.p >= .75].y.mean())},
              "p<=0.25": {"n": int((te.p <= .25).sum()), "actual_good_rate": float(te[te.p <= .25].y.mean())},
              "0.25<p<0.75": {"n": int(((te.p > .25) & (te.p < .75)).sum()), "actual_good_rate": float(te[(te.p > .25) & (te.p < .75)].y.mean())}}
print(R["hc_total"], R["hc_pct"], R["hc_fp"], R["hc_fn"]); print(trait_df.round(3))

# %% [markdown]
# ## 2.4 Fairness trade-off (sensitive variable: personal status & sex → Male vs Female)

# %%
male = te.male.values; ths = np.round(np.arange(0.05, 0.96, 0.10), 2)
def safe(a, b): return float(a/b) if b not in (0, 0.0) and not np.isnan(b) else np.nan
def rates(g, t):
    yh = (pB[g] >= t).astype(int); yy = yte[g]
    sel = yh.mean()
    tpr = yh[yy == 1].mean() if (yy == 1).any() else np.nan
    fpr = yh[yy == 0].mean() if (yy == 0).any() else np.nan
    ppv = yy[yh == 1].mean() if (yh == 1).any() else np.nan
    return sel, tpr, fpr, ppv
sweep = []
for t in ths:
    sA, tA, fA, pA_ = rates(male, t); sB, tB, fB, pB_ = rates(~male, t)
    dp = safe(sA, sB); r_t, r_f = safe(tA, tB), safe(fA, fB)
    eo = np.nanmin([r_t, r_f]) if not (np.isnan(r_t) and np.isnan(r_f)) else np.nan
    pp = safe(pA_, pB_); yh_all = (pB >= t)
    sweep.append(dict(t=float(t), DP=dp, EO=float(eo), PP=pp, TPR_ratio=r_t, FPR_ratio=r_f, sel_M=sA, sel_F=sB, TPR_M=tA, TPR_F=tB,
                      FPR_M=fA, FPR_F=fB, PPV_M=pA_, PPV_F=pB_, acc=float((yh_all == yte).mean()),
                      uci_cost=int(5*((yh_all) & (yte == 0)).sum() + ((~yh_all) & (yte == 1)).sum()), approved=int(yh_all.sum()), good_rejected=int(((~yh_all) & (yte == 1)).sum()), bad_approved=int((yh_all & (yte == 0)).sum())))
sw = pd.DataFrame(sweep); sw.to_csv("outputs/fairness_sweep.csv", index=False)
R["sweep"] = sweep; R["n_male_test"], R["n_female_test"] = int(male.sum()), int((~male).sum())
R["test_good_by_sex"] = {"M": int(yte[male].sum()), "F": int(yte[~male].sum())}
print(sw[["t","DP","EO","PP","acc","approved","good_rejected","bad_approved"]].round(3))

# Trade-off cost, computed by rule. For each metric pair (X, Y) look at every move from threshold i to j (t <= 0.85, the
# degenerate 0.95 point excluded) that brings X closer to 1 (gain Z%) while pushing Y further from 1 (loss W%). Distances
# are |metric - 1| because 1.0 is the ideal value. The move with the largest Z is reported; "none" if X and Y never conflict.
elig = sw[sw.t <= 0.85].reset_index(drop=True)
def tradeoff(X, Y):
    best = None
    for i in range(len(elig)):
        for j in range(len(elig)):
            dx0, dx1 = abs(elig.loc[i, X]-1), abs(elig.loc[j, X]-1); dy0, dy1 = abs(elig.loc[i, Y]-1), abs(elig.loc[j, Y]-1)
            if dx1 < dx0 and dy1 > dy0 and dx0 > 0 and dy0 >= 0:
                Z = 100*(dx0-dx1)/dx0; W = 100*(dy1-dy0)/max(dy0, 1e-9) if dy0 > 1e-9 else np.nan
                if np.isnan(W): continue
                cand = dict(X=X, Y=Y, t_from=float(elig.t[i]), t_to=float(elig.t[j]), X_from=float(elig.loc[i, X]), X_to=float(elig.loc[j, X]),
                            Y_from=float(elig.loc[i, Y]), Y_to=float(elig.loc[j, Y]), Z_pct=float(Z), W_pct=float(W), cost=float(W/Z))
                if best is None or Z > best["Z_pct"]: best = cand
    return best
R["tradeoffs"] = {f"{x}_vs_{y}": tradeoff(x, y) for x, y in [("DP", "EO"), ("DP", "PP"), ("EO", "PP"), ("PP", "DP"), ("PP", "EO"), ("EO", "DP")]}
print({k: (None if v is None else {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in R["tradeoffs"].items()})
R["best_by_metric"] = {m: float(elig.loc[(elig[m]-1).abs().idxmin(), "t"]) for m in ["DP", "EO", "PP"]}

fig, ax = plt.subplots(figsize=(7.0, 5.2))
ax.axhline(1, color="grey", lw=.8, ls="--"); ax.axvline(1, color="grey", lw=.8, ls="--")
ins = sw[(sw.DP <= 1.5) & (sw.EO <= 1.5)]
ax.plot(ins.DP, ins.EO, color="#9aa9b8", lw=1, zorder=2)
ax.scatter(ins.DP, ins.EO, c="#1f3b57", s=40, zorder=3)
offs = {0.05: (-26, -14), 0.15: (-26, 4), 0.25: (-4, -15), 0.35: (-26, -14), 0.45: (-4, 8), 0.55: (6, -4), 0.65: (6, -4), 0.75: (6, 4), 0.85: (6, -12)}
for _, r in ins.iterrows(): ax.annotate(f"t={r.t:.2f}", (r.DP, r.EO), textcoords="offset points", xytext=offs.get(round(r.t, 2), (5, 5)), fontsize=8)
ax.set_xlim(0, 1.5); ax.set_ylim(0, 1.5); ax.set_xlabel("Demographic Parity ratio (Male / Female)"); ax.set_ylabel("Equalized Odds ratio = min(TPR ratio, FPR ratio)")
ax.set_title("Fairness trade-off across decision thresholds"); plt.tight_layout(); plt.savefig("outputs/fig_fairness.png", dpi=200); plt.close()
R["points_outside_chart"] = [dict(t=float(r.t), DP=float(r.DP), EO=float(r.EO)) for _, r in sw.iterrows() if not (r.DP <= 1.5 and r.EO <= 1.5)]

# %% [markdown]
# ## 2.5 Topic lens: where does a human reviewer still have room to decide?
# Extension for the chosen topic (*does AI reduce human autonomy in decision making?*). It reuses the fitted logistic model only.

# %%
bands = {"confident reject (p<=0.25)": te.p <= .25, "uncertain (0.25<p<0.75)": (te.p > .25) & (te.p < .75), "confident approve (p>=0.75)": te.p >= .75}
band_rows = []
for k, m in bands.items():
    d = te[m]; pred = (d.p >= .5).astype(int)
    band_rows.append(dict(band=k, n=int(len(d)), pct=100*len(d)/len(te), model_acc=float((pred == d.y).mean()) if len(d) else np.nan,
                          actual_good_rate=float(d.y.mean()) if len(d) else np.nan))
R["autonomy_bands"] = band_rows
R["autonomy_flip_035_065"] = int(((te.p >= .35) != (te.p >= .65)).sum())
R["autonomy_flip_035_065_pct"] = 100*R["autonomy_flip_035_065"]/len(te)
R["autonomy_flip_ols_vs_logit_pct"] = R["mismatch_pct"]
R["autonomy_flip_full_vs_reduced_pct"] = 100*R["ovb_decision_flips"]/len(te)
# if a reviewer defers to every confident score: how many wrong final decisions come from the confident band?
conf = te[(te.p >= .75) | (te.p <= .25)]
R["autonomy_conf_n"], R["autonomy_conf_wrong"] = int(len(conf)), int(len(wrong))
R["autonomy_conf_wrong_rate"] = float(len(wrong)/len(conf))
R["autonomy_uncertain_wrong_rate"] = float(1-band_rows[1]["model_acc"])
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.hist([te[te.y == 1].p, te[te.y == 0].p], bins=np.linspace(0, 1, 21), stacked=True, color=["#2a6f97", "#c1666b"], label=["actual Good", "actual Bad"])
ax.axvspan(.25, .75, color="grey", alpha=.12); ax.set_xlabel("Logistic P(approve)"); ax.set_ylabel("test applicants"); ax.legend(frameon=False)
ax.set_title("Shaded band = scores where a reviewer has genuine room to decide"); plt.tight_layout(); plt.savefig("outputs/fig_autonomy.png", dpi=200); plt.close()
print(pd.DataFrame(band_rows).round(3))

# %%
# Row-level predictions (used for the worked two-applicant example in Part 3, Q6). `orig_row` = 0-based row in german.data
pd.DataFrame({"orig_row": te_df.index, "sex": np.where(te.male, "Male", "Female"), "age": te.age.values, "credit_amount": te.credit_amount.values,
              "duration": te.duration.values, "checking_status": te.checking_status.values, "purpose": te.purpose.values,
              "y_good": yte, "p_logit": pB, "p_ols": pA}).to_csv("outputs/test_predictions.csv", index=False)
R["good_rate_by_sex_test"] = {"M": float(yte[male].mean()), "F": float(yte[~male].mean())}
with open("outputs/results.json", "w") as f: json.dump(R, f, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else (o.item() if hasattr(o, "item") else str(o)))
print("saved outputs/results.json")
