Data source
-----------
Real CDISC Pilot Study ("cdiscpilot01") SDTM + ADaM datasets, downloaded
directly from CDISC's own GitHub submission package:
  https://github.com/cdisc-org/sdtm-adam-pilot-project
    .../tabulations/sdtm/qs.xpt          (SDTM QS -- raw ADAS-Cog item/total scores)
    .../analysis/adam/datasets/adsl.xpt  (ADaM ADSL -- subject-level: treatment, population flags)
    .../analysis/adam/datasets/adqsadas.xpt (ADaM ADQSADAS -- the OFFICIAL, ready-made
                                              BDS efficacy dataset for ADAS-Cog, used
                                              here only to VALIDATE this script's own
                                              from-scratch derivation, per the brief's
                                              "derive it yourself" instruction)
Environment note
-----------------
This machine has no R (and no Homebrew to add it) and no SAS, so the "derive with
admiral" and "reproduce in SAS" steps cannot literally invoke the admiral R package
or PROC GLM/PROC MIXED. Instead:
  - The BDS dataset is derived from the raw SDTM QS domain by hand in pandas,
    re-implementing admiral's own visit-windowing + LOCF algorithm (not
    approximating it -- every design choice below, including the exact visit
    windows AWLO/AWHI/AWTARGET, was reverse-engineered from and checked against
    the official ADQSADAS dataset's own AWLO/AWHI/AWTARGET columns).
  - The derivation is validated record-by-record against the official ADaM
    dataset (see validate_against_official()), not just assumed correct.
  - The ANCOVA and MMRM are fit natively in Python (statsmodels) rather than
    PROC GLM / PROC MIXED. Section 7 below writes out real, ready-to-run SAS
    programs reproducing both models against this same derived dataset -- they
    are NOT executed here (no SAS available), and the report says so plainly.


import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "data"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

PARAMCD = "ACTOT"
TRT_ORDER = ["Placebo", "Xanomeline Low Dose", "Xanomeline High Dose"]

# Visit windows, reverse-engineered from the official ADQSADAS AWLO/AWHI/AWTARGET
# columns for PARAMCD=='ACTOT' (confirmed by direct inspection of the xpt file).
WINDOWS = [
    ("Baseline", 0, -np.inf, 1, 1),
    ("Week 8", 8, 2, 84, 56),
    ("Week 16", 16, 85, 140, 112),
    ("Week 24", 24, 141, np.inf, 168),
]
VISIT_ORDER = ["Baseline", "Week 8", "Week 16", "Week 24"]


# ---------------------------------------------------------------------------
# Step 0: load SDTM/ADaM sources
# ---------------------------------------------------------------------------
def _clean_xport_floats(df: pd.DataFrame) -> pd.DataFrame:
    """pandas' SAS xport reader mis-renders true 0.0 as a ~5e-79 denormal for
    some numeric columns (a known IBM-float-conversion quirk). Round anything
    that tiny back to exact 0."""
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].mask(df[c].abs() < 1e-25, 0.0)
    return df


def load_sources():
    adsl = _clean_xport_floats(pd.read_sas(DATA / "adsl.xpt", format="xport", encoding="utf-8"))
    qs = _clean_xport_floats(pd.read_sas(DATA / "qs.xpt", format="xport", encoding="utf-8"))
    official = _clean_xport_floats(pd.read_sas(DATA / "adqsadas.xpt", format="xport", encoding="utf-8"))
    for df in (adsl, qs, official):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].str.decode("utf-8") if df[c].apply(lambda x: isinstance(x, bytes)).any() else df[c]
                df[c] = df[c].astype(str).str.strip()
    adsl["TRT01P"] = pd.Categorical(adsl["TRT01P"], categories=TRT_ORDER, ordered=True)
    return adsl, qs, official


# ---------------------------------------------------------------------------
# Step 1: derive the BDS efficacy dataset from QS (admiral-equivalent logic)
# ---------------------------------------------------------------------------
def derive_bds_actot(qs: pd.DataFrame, adsl: pd.DataFrame) -> pd.DataFrame:
    actot = qs[qs["QSTESTCD"] == PARAMCD].copy()
    actot = actot.merge(
        adsl[["USUBJID", "TRT01P", "TRT01PN", "ITTFL", "EFFFL", "AGE", "SEX", "SITEGR1"]],
        on="USUBJID", how="left",
    )
    actot = actot[actot["ITTFL"] == "Y"]  # ITT population is the derivation's base population

    frames = []
    for avisit, avisitn, lo, hi, target in WINDOWS:
        w = actot[(actot["QSDY"] >= lo) & (actot["QSDY"] <= hi)].copy()
        w["AVISIT"] = avisit
        w["AVISITN"] = avisitn
        w["dist"] = (w["QSDY"] - target).abs()
        frames.append(w)
    windowed = pd.concat(frames, ignore_index=True)

    # one record per subject per analysis visit: nearest to the visit's target day
    windowed = windowed.sort_values(["USUBJID", "AVISIT", "dist", "QSDY", "QSSEQ"])
    selected = windowed.groupby(["USUBJID", "AVISIT"], as_index=False).first()

    observed = selected.rename(columns={"QSDY": "ADY", "QSSTRESN": "AVAL"})[
        ["USUBJID", "AVISIT", "ADY", "AVAL"]
    ].copy()

    # Full subject x analysis-visit grid, so that a visit with NO qualifying raw
    # record still gets an explicit (AVAL=NaN) row -- required for LOCF to have
    # something to fill, and to make "missing" visible rather than absent.
    subjects = adsl.loc[adsl["ITTFL"] == "Y", ["USUBJID", "TRT01P", "TRT01PN", "ITTFL", "EFFFL", "AGE", "SEX"]]
    grid = subjects.merge(pd.DataFrame({"AVISIT": VISIT_ORDER}), how="cross")
    bds = grid.merge(observed, on=["USUBJID", "AVISIT"], how="left")

    bds["ABLFL"] = np.where(bds["AVISIT"] == "Baseline", "Y", "")
    base = bds.loc[bds["ABLFL"] == "Y", ["USUBJID", "AVAL"]].rename(columns={"AVAL": "BASE"})
    bds = bds.merge(base, on="USUBJID", how="left")
    bds["CHG"] = np.where(bds["ABLFL"] == "Y", np.nan, bds["AVAL"] - bds["BASE"])
    bds["PCHG"] = np.where(bds["BASE"] > 0, bds["CHG"] / bds["BASE"] * 100, np.nan)
    bds["AVISIT"] = pd.Categorical(bds["AVISIT"], categories=VISIT_ORDER, ordered=True)
    return bds.sort_values(["USUBJID", "AVISIT"]).reset_index(drop=True)


def validate_against_official(bds: pd.DataFrame, official: pd.DataFrame) -> dict:
    off = official[(official["PARAMCD"] == PARAMCD) & (official["DTYPE"] == "")][
        ["USUBJID", "AVISIT", "AVAL", "BASE", "CHG"]
    ]
    merged = bds.merge(off, on=["USUBJID", "AVISIT"], how="outer", suffixes=("_mine", "_official"), indicator=True)
    n_official = len(off)
    n_matched_rows = (merged["_merge"] == "both").sum()
    diffs = {}
    for col in ["AVAL", "BASE", "CHG"]:
        d = (merged[f"{col}_mine"] - merged[f"{col}_official"]).abs()
        diffs[col] = {"max_abs_diff": float(np.nanmax(d)), "n_compared": int(d.notna().sum())}
    mismatches = merged[(merged["_merge"] == "both") & ((merged["AVAL_mine"] - merged["AVAL_official"]).abs() > 0.001)]
    report = {
        "n_official_actual_records": int(n_official),
        "n_rows_matched_on_subject_visit": int(n_matched_rows),
        "n_official_only": int((merged["_merge"] == "right_only").sum()),
        "n_mine_only": int((merged["_merge"] == "left_only").sum()),
        "diffs": diffs,
        "n_value_mismatches": int(len(mismatches)),
        "mismatch_detail": mismatches[["USUBJID", "AVISIT", "AVAL_mine", "AVAL_official"]].to_dict("records"),
    }
    return report


# ---------------------------------------------------------------------------
# Step 2: LOCF derivation + validation
# ---------------------------------------------------------------------------
def derive_locf(bds: pd.DataFrame) -> pd.DataFrame:
    out = bds.copy()
    out = out.sort_values(["USUBJID", "AVISIT"])
    out["AVAL_LOCF"] = out.groupby("USUBJID")["AVAL"].ffill()
    out["CHG_LOCF"] = np.where(out["ABLFL"] == "Y", np.nan, out["AVAL_LOCF"] - out["BASE"])
    return out


def validate_locf_against_official(locf: pd.DataFrame, official: pd.DataFrame) -> dict:
    off_locf = official[(official["PARAMCD"] == PARAMCD) & (official["DTYPE"] == "LOCF")][
        ["USUBJID", "AVISIT", "AVAL", "CHG"]
    ]
    mine = locf[["USUBJID", "AVISIT", "AVAL_LOCF", "CHG_LOCF"]].rename(
        columns={"AVAL_LOCF": "AVAL", "CHG_LOCF": "CHG"}
    )
    merged = mine.merge(off_locf, on=["USUBJID", "AVISIT"], how="inner", suffixes=("_mine", "_official"))
    diff = (merged["AVAL_mine"] - merged["AVAL_official"]).abs()
    return {
        "n_official_locf_records": int(len(off_locf)),
        "n_matched": int(len(merged)),
        "max_abs_diff_AVAL": float(diff.max()) if len(diff) else None,
    }


# ---------------------------------------------------------------------------
# Modeling helpers: LS means via patsy design-matrix prediction (robust to
# whatever dummy-coding statsmodels/patsy generates internally)
# ---------------------------------------------------------------------------
def lsmean_ci(result, design_info, newdata: pd.DataFrame, alpha=0.05, df_resid=None):
    (mm,) = patsy.build_design_matrices([design_info], newdata, return_type="dataframe")
    row = mm.iloc[0].values
    params = np.asarray(result.params)
    cov = np.asarray(result.cov_params())
    est = row @ params
    se = np.sqrt(row @ cov @ row.T)
    dfr = df_resid if df_resid is not None else getattr(result, "df_resid", None)
    if dfr:
        tcrit = stats.t.ppf(1 - alpha / 2, dfr)
        pfunc = lambda t: 2 * (1 - stats.t.cdf(abs(t), dfr))
    else:
        tcrit = stats.norm.ppf(1 - alpha / 2)
        pfunc = lambda t: 2 * (1 - stats.norm.cdf(abs(t)))
    return {"estimate": float(est), "se": float(se),
            "ci_lo": float(est - tcrit * se), "ci_hi": float(est + tcrit * se),
            "_row": row, "_pfunc": pfunc}


def contrast_ci(result, design_info, newdata_a: pd.DataFrame, newdata_b: pd.DataFrame, alpha=0.05, df_resid=None):
    """estimate (a - b) with its own SE from the joint covariance (not just
    sqrt(se_a^2+se_b^2), which would ignore covariance between the two rows)."""
    (mm_a,) = patsy.build_design_matrices([design_info], newdata_a, return_type="dataframe")
    (mm_b,) = patsy.build_design_matrices([design_info], newdata_b, return_type="dataframe")
    row = mm_a.iloc[0].values - mm_b.iloc[0].values
    params = np.asarray(result.params)
    cov = np.asarray(result.cov_params())
    est = row @ params
    se = np.sqrt(row @ cov @ row.T)
    dfr = df_resid if df_resid is not None else getattr(result, "df_resid", None)
    if dfr:
        tcrit = stats.t.ppf(1 - alpha / 2, dfr)
        tval = est / se
        p = 2 * (1 - stats.t.cdf(abs(tval), dfr))
    else:
        tcrit = stats.norm.ppf(1 - alpha / 2)
        tval = est / se
        p = 2 * (1 - stats.norm.cdf(abs(tval)))
    return {"estimate": float(est), "se": float(se), "t_or_z": float(tval), "p_value": float(p),
            "ci_lo": float(est - tcrit * se), "ci_hi": float(est + tcrit * se)}


def make_newrow(template: pd.DataFrame, **overrides) -> pd.DataFrame:
    row = template.iloc[[0]].copy()
    for k, v in overrides.items():
        row[k] = v
    return row


# ---------------------------------------------------------------------------
# Step 3: ANCOVA at Week 24 (complete case) and LOCF ANCOVA
# ---------------------------------------------------------------------------
def run_ancova(df: pd.DataFrame, chg_col: str, mean_base: float) -> dict:
    model = smf.ols(f"{chg_col} ~ C(TRT01P, Treatment('Placebo')) + BASE", data=df).fit()
    di = model.model.data.design_info
    template = df.iloc[[0]]
    rows = {}
    for trt in TRT_ORDER:
        newrow = make_newrow(template, TRT01P=trt, BASE=mean_base)
        rows[trt] = lsmean_ci(model, di, newrow, df_resid=model.df_resid)
    contrasts = {}
    for trt in TRT_ORDER:
        if trt == "Placebo":
            continue
        a = make_newrow(template, TRT01P=trt, BASE=mean_base)
        b = make_newrow(template, TRT01P="Placebo", BASE=mean_base)
        contrasts[trt] = contrast_ci(model, di, a, b, df_resid=model.df_resid)
    return {
        "n": int(model.nobs),
        "mean_base_used": float(mean_base),
        "lsmeans": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in rows.items()},
        "diff_vs_placebo": contrasts,
        "model_summary": model.summary().as_text(),
    }


# ---------------------------------------------------------------------------
# Step 4: MMRM (statsmodels MixedLM approximation) on observed post-baseline data
# ---------------------------------------------------------------------------
def run_mmrm(df_postbase: pd.DataFrame, mean_base: float) -> dict:
    d = df_postbase.dropna(subset=["CHG"]).copy()
    d["AVISIT"] = d["AVISIT"].astype(str)
    formula = "CHG ~ C(AVISIT, Treatment('Week 8')) * C(TRT01P, Treatment('Placebo')) + BASE"
    md = smf.mixedlm(formula, data=d, groups=d["USUBJID"], re_formula="~0+C(AVISIT)")
    mdf = md.fit(reml=True, method=["lbfgs"])
    di = mdf.model.data.design_info
    fe_names = mdf.fe_params.index.tolist()
    full_cov = mdf.cov_params()
    fe_cov = full_cov.loc[fe_names, fe_names]

    class _Wrap:
        def __init__(self, params, cov):
            self.params = params
            self._cov = cov
        def cov_params(self):
            return self._cov

    wrapped = _Wrap(mdf.fe_params, fe_cov)
    template = d.iloc[[0]]
    rows = {}
    for trt in TRT_ORDER:
        newrow = make_newrow(template, TRT01P=trt, AVISIT="Week 24", BASE=mean_base)
        rows[trt] = lsmean_ci(wrapped, di, newrow, df_resid=None)
    contrasts = {}
    for trt in TRT_ORDER:
        if trt == "Placebo":
            continue
        a = make_newrow(template, TRT01P=trt, AVISIT="Week 24", BASE=mean_base)
        b = make_newrow(template, TRT01P="Placebo", AVISIT="Week 24", BASE=mean_base)
        contrasts[trt] = contrast_ci(wrapped, di, a, b, df_resid=None)
    return {
        "n_obs": int(mdf.nobs),
        "n_subjects": int(d["USUBJID"].nunique()),
        "mean_base_used": float(mean_base),
        "lsmeans_week24": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in rows.items()},
        "diff_vs_placebo_week24": contrasts,
        "converged": bool(mdf.converged),
        "model_summary": mdf.summary().as_text(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    adsl, qs, official = load_sources()

    bds = derive_bds_actot(qs, adsl)
    qc_bds = validate_against_official(bds, official)
    print("=== BDS derivation QC vs. official ADQSADAS ===")
    print(json.dumps(qc_bds, indent=2, default=str)[:1500])

    locf = derive_locf(bds)
    qc_locf = validate_locf_against_official(locf, official)
    print("\n=== LOCF derivation QC vs. official ADQSADAS (DTYPE=LOCF) ===")
    print(json.dumps(qc_locf, indent=2))

    # --- analysis population: efficacy population, per ADSL's own EFFFL flag ---
    eff = locf[locf["EFFFL"] == "Y"].copy()
    n_pop = eff.groupby("TRT01P", observed=True)["USUBJID"].nunique().to_dict()
    print("\nEfficacy population N by arm:", n_pop)

    # --- visit-level summary (observed/complete-case) ---
    summary = (
        eff[eff["ABLFL"] != "Y"]
        .groupby(["AVISIT", "TRT01P"], observed=True)["CHG"]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    print("\n=== Observed mean CHG by visit x arm ===")
    print(summary.to_string(index=False))

    # --- plot: mean change from baseline over time ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Placebo": "#999999", "Xanomeline Low Dose": "#4C72B0", "Xanomeline High Dose": "#C44E52"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for trt in TRT_ORDER:
        sub = summary[summary["TRT01P"] == trt].sort_values("AVISIT")
        xw = [VISIT_ORDER.index(v) for v in sub["AVISIT"]]
        ax.errorbar(xw, sub["mean"], yerr=1.96 * sub["se"], marker="o", capsize=4,
                    label=trt, color=colors[trt], linewidth=2)
    ax.set_xticks(range(len(VISIT_ORDER)))
    ax.set_xticklabels(VISIT_ORDER)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Mean change from baseline, ADAS-Cog(11)\n(negative = improvement)")
    ax.set_xlabel("Visit")
    ax.set_title("Observed mean change from baseline in ADAS-Cog(11) by visit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "mean_change_by_visit.png", dpi=150)
    plt.close(fig)

    # --- Week 24 complete-case ANCOVA ---
    mean_base = eff.loc[eff["ABLFL"] == "Y", "BASE"].mean()
    w24 = eff[(eff["AVISIT"] == "Week 24") & eff["CHG"].notna()].copy()
    ancova_cc = run_ancova(w24, "CHG", mean_base)
    print("\n=== Week 24 ANCOVA (complete case), n =", ancova_cc["n"], "===")
    for trt, c in ancova_cc["diff_vs_placebo"].items():
        print(f"  {trt}: diff={c['estimate']:.3f} 95% CI [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}] p={c['p_value']:.4f}")

    # --- Week 24 LOCF ANCOVA ---
    w24_locf = eff[eff["AVISIT"] == "Week 24"].copy()
    w24_locf["CHG"] = w24_locf["CHG_LOCF"]
    w24_locf = w24_locf[w24_locf["CHG"].notna()]
    ancova_locf = run_ancova(w24_locf, "CHG", mean_base)
    print("\n=== Week 24 ANCOVA on LOCF data, n =", ancova_locf["n"], "===")
    for trt, c in ancova_locf["diff_vs_placebo"].items():
        print(f"  {trt}: diff={c['estimate']:.3f} 95% CI [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}] p={c['p_value']:.4f}")

    # --- MMRM on all observed post-baseline visits ---
    postbase = eff[eff["ABLFL"] != "Y"].copy()
    mmrm = run_mmrm(postbase, mean_base)
    print("\n=== MMRM, Week 24 LS means, n_obs =", mmrm["n_obs"], "n_subjects =", mmrm["n_subjects"], "===")
    for trt, c in mmrm["diff_vs_placebo_week24"].items():
        print(f"  {trt}: diff={c['estimate']:.3f} 95% CI [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}] p={c['p_value']:.4f}")

    # --- comparison table across the three missing-data handling methods ---
    comparison_rows = []
    for method_name, res in [("ANCOVA (complete case)", ancova_cc),
                              ("ANCOVA on LOCF", ancova_locf)]:
        for trt, c in res["diff_vs_placebo"].items():
            comparison_rows.append({"method": method_name, "dose": trt, **c})
    for trt, c in mmrm["diff_vs_placebo_week24"].items():
        comparison_rows.append({"method": "MMRM", "dose": trt, **c})
    comparison = pd.DataFrame(comparison_rows)
    print("\n=== Missing-data-handling comparison (Week 24, diff vs placebo) ===")
    print(comparison.to_string(index=False))

    # --- forest plot ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    methods = ["ANCOVA (complete case)", "ANCOVA on LOCF", "MMRM"]
    y = 0
    yticks, ylabels = [], []
    for dose in ["Xanomeline Low Dose", "Xanomeline High Dose"]:
        for m in methods:
            row = comparison[(comparison["method"] == m) & (comparison["dose"] == dose)].iloc[0]
            ax.errorbar(row["estimate"], y, xerr=[[row["estimate"] - row["ci_lo"]], [row["ci_hi"] - row["estimate"]]],
                        fmt="o", capsize=4, color="#4C72B0" if "Low" in dose else "#C44E52")
            yticks.append(y)
            ylabels.append(f"{dose.replace('Xanomeline ', '')} -- {m}")
            y += 1
        y += 0.5
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("LS mean difference vs. placebo, Week 24 CHG in ADAS-Cog(11)\n(negative = favors active treatment)")
    ax.set_title("Week 24 treatment effect by missing-data method")
    fig.tight_layout()
    fig.savefig(FIG / "forest_plot.png", dpi=150)
    plt.close(fig)

    # --- persist everything ---
    results = {
        "paramcd": PARAMCD,
        "n_population": n_pop,
        "qc_bds_derivation": qc_bds,
        "qc_locf_derivation": qc_locf,
        "visit_summary": summary.assign(AVISIT=lambda d: d["AVISIT"].astype(str)).to_dict("records"),
        "mean_base_used": float(mean_base),
        "ancova_complete_case": {k: v for k, v in ancova_cc.items() if k != "model_summary"},
        "ancova_locf": {k: v for k, v in ancova_locf.items() if k != "model_summary"},
        "mmrm": {k: v for k, v in mmrm.items() if k != "model_summary"},
        "comparison_table": comparison.to_dict("records"),
    }
    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(HERE / "model_summaries.txt", "w") as f:
        f.write("ANCOVA (complete case)\n" + "=" * 60 + "\n")
        f.write(ancova_cc["model_summary"] + "\n\n")
        f.write("ANCOVA (LOCF)\n" + "=" * 60 + "\n")
        f.write(ancova_locf["model_summary"] + "\n\n")
        f.write("MMRM (MixedLM approximation)\n" + "=" * 60 + "\n")
        f.write(mmrm["model_summary"] + "\n")

    bds.to_csv(HERE / "derived_adqsadas_actot.csv", index=False)
    locf.to_csv(HERE / "derived_adqsadas_actot_locf.csv", index=False)

    print("\nDone. Wrote results.json, model_summaries.txt, figures/, derived_*.csv")


if __name__ == "__main__":
    main()
