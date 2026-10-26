# Alzheimer-s-Trial-Analysis-CDISC-ADaM-Efficacy-
Did xanomeline improve cognition (ADAS-Cog(11) total score) compared with placebo at Week 24, in the CDISC Pilot Alzheimer's trial — and does the answer change depending on how dropout (missing follow-up visits) is handled statistically?

## 1. Use Case

### 1.1 The business/clinical question

**Did xanomeline improve cognition (ADAS-Cog(11) total score) compared with placebo at Week 24,
in the CDISC Pilot Alzheimer's trial — and does the answer change depending on how dropout
(missing follow-up visits) is handled statistically?**

The CDISC Pilot Study (`cdiscpilot01`) is a randomized, placebo-controlled Alzheimer's trial
with three arms:

| Arm | N (efficacy population) |
|---|---|
| Placebo | 79 |
| Xanomeline Low Dose | 81 |
| Xanomeline High Dose | 74 |

Cognition is measured by **ADAS-Cog(11)** — an 11-item, 0–70 point clinician-administered scale.
**Lower = better cognition.** Therefore a **negative** change from baseline (`CHG`) = improvement,
positive = decline. This sign convention is the single most common source of misreading in this
project.

### 1.2 What the script actually does (5 jobs)

1. **Derives an ADaM BDS efficacy dataset from raw SDTM data** — i.e. it reimplements, in pandas,
   what the R package `admiral` would normally do (visit windowing, baseline flagging,
   change-from-baseline, LOCF).
2. **Validates that derivation record-by-record** against the *official* CDISC ADaM dataset
   (`adqsadas.xpt`) that ships in the same submission package — so correctness is *proven*,
   not assumed.
3. **Summarizes** observed mean change from baseline by visit × treatment arm, and plots it.
4. **Fits three different Week-24 treatment-effect models** — complete-case ANCOVA, LOCF ANCOVA,
   and MMRM — to quantify how sensitive the conclusion is to missing-data handling.
5. **Persists everything** (JSON results, model summaries, two CSVs, two PNG figures) for the
   downstream report generators and SAS cross-check scripts.
## 2. Data Contract (Inputs)

All three files live in `./data/` and are SAS transport (`.xpt`) files downloaded from
CDISC's own GitHub submission package:
`https://github.com/cdisc-org/sdtm-adam-pilot-project` →
`updated-pilot-submission-package/900172/m5/datasets/cdiscpilot01/`

| File | Standard | Role | Key columns used |
|---|---|---|---|
| `qs.xpt` | SDTM **QS** (Questionnaires) | **Primary input.** Raw ADAS-Cog item + total scores. 121,749 rows, 6 different questionnaires stacked. | `USUBJID`, `QSTESTCD`, `QSSTRESN`, `QSDY`, `QSSEQ`, `VISIT`, `VISITNUM`, `QSCAT` |
| `adsl.xpt` | ADaM **ADSL** (Subject-Level) | Subject-level merge source: treatment arm, population flags, demographics. 254 rows (1/subject). | `USUBJID`, `TRT01P`, `TRT01PN`, `ITTFL`, `EFFFL`, `AGE`, `SEX`, `SITEGR1` |
| `adqsadas.xpt` | ADaM **ADQSADAS** (BDS) | **Validation reference ONLY.** The official, ready-made ADAS-Cog BDS dataset. Never used to produce results — only to check the script's own derivation. | `USUBJID`, `PARAMCD`, `AVISIT`, `AVAL`, `BASE`, `CHG`, `DTYPE`, `ADY`, `AWLO`, `AWHI`, `AWTARGET` |

### Critical input facts

- `QSTESTCD == "ACTOT"` is the **ADAS-Cog(11) total score** (the derived total, `QSDRVFL='Y'`).
  `ACITM01`–`ACITM14` are the individual items — **not used** by this pipeline.
- `QSDY` (SDTM study day) was **verified to equal** the official ADaM `ADY` exactly
  (max abs diff = 0 across all 799 matched records), and `QSSTRESN` equals official `AVAL` exactly.
  This is why the script can window on `QSDY` directly.
- In `adqsadas.xpt`, `DTYPE == ""` means an **actual observed** record; `DTYPE == "LOCF"` means an
  **imputed** record. The script validates against both groups separately.
- Raw QS contains off-schedule visits (`WEEK 2`, `WEEK 4`, `WEEK 6`, `WEEK 12`, `WEEK 20`,
  `RETRIEVAL`, `AMBUL ECG REMOVAL`). The official ADaM has only 4 `AVISIT` values — the windowing
  logic is what collapses the former into the latter.


## Headline result

All three Week-24 estimators (complete-case ANCOVA, ANCOVA on LOCF, MMRM) point
the same direction for the high dose vs. placebo, but the LOCF estimate is
visibly larger in magnitude with a deceptively narrow CI -- the classic LOCF
failure mode of treating an imputed value as if it were observed. None of the
three reaches conventional significance in this ~250-subject teaching dataset
(all p > 0.05), which is an expected, honest result for a small trial, not
evidence the drug does nothing. Full efficacy table and forest plot in
`project4_adascog_efficacy_report.docx`.

## Key limitations

`cdiscpilot01` is CDISC's public teaching dataset, not a real pivotal trial.
The MMRM is a statsmodels (MixedLM) approximation of a true PROC MIXED
REPEATED/UN model -- close, not numerically identical; the real SAS programs
are provided but unexecuted. Two of ~1,040 derived values (0.2%) differ from
the official ADaM dataset at a single visit-windowing tie-break edge case.
