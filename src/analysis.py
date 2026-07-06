"""
analysis.py -- regression and effect-size analysis of proxy feature impact
on LLM hiring scores. Substance is held constant by experimental design, so
any score variance explained by proxy features here is Goodhart-style bias,
not a real competency signal.

Importable by both a CLI script and the Streamlit dashboard -- no duplicated
logic between the two.
"""

import json
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
VARIANTS_PATH = ROOT / "data" / "resume_variants.json"
RESULTS_PATH = ROOT / "results" / "scores.csv"

PROXY_COLUMNS = ["university_tier", "keyword_framing", "employment_gap", "years_framing"]


@dataclass
class RegressionResult:
    model: "sm.regression.linear_model.RegressionResultsWrapper"
    coef_table: pd.DataFrame
    r_squared: float
    n_obs: int


def load_merged_data(provider: str | None = None) -> pd.DataFrame:
    """Merges variant proxy labels with LLM scores into one dataframe."""
    variants = pd.DataFrame(json.loads(VARIANTS_PATH.read_text()))
    scores = pd.read_csv(RESULTS_PATH)

    if provider:
        scores = scores[scores["provider"] == provider]

    merged = scores.merge(variants, on="variant_id", how="left")
    return merged


def run_regression(df: pd.DataFrame) -> RegressionResult:
    """
    OLS: score ~ proxy features (as categorical dummies).
    base_resume_id is included as a fixed effect to absorb any residual
    cross-base variance, so coefficients reflect within-base proxy effects.
    """
    formula = (
        "score ~ C(university_tier) + C(keyword_framing) + C(employment_gap) "
        "+ C(years_framing) + C(base_resume_id)"
    )
    model = smf.ols(formula=formula, data=df).fit()

    coef_table = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "p_value": model.pvalues.values,
        "ci_lower": model.conf_int()[0].values,
        "ci_upper": model.conf_int()[1].values,
    })
    # Drop intercept and base_resume fixed effects from the "proxy effects" view
    proxy_view = coef_table[
        ~coef_table["term"].str.contains("Intercept|base_resume_id")
    ].reset_index(drop=True)

    return RegressionResult(
        model=model, coef_table=proxy_view, r_squared=model.rsquared, n_obs=int(model.nobs)
    )


def effect_size_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, interpretable companion to the regression: mean score for each
    level of each proxy feature, holding nothing else fixed (marginal means).
    Useful for the dashboard's plain-language bar chart.
    """
    rows = []
    for col in PROXY_COLUMNS:
        grouped = df.groupby(col)["score"].agg(["mean", "std", "count"])
        for level, stats in grouped.iterrows():
            rows.append({
                "proxy_feature": col,
                "level": level,
                "mean_score": stats["mean"],
                "std_score": stats["std"],
                "n": int(stats["count"]),
            })
    return pd.DataFrame(rows)


def plain_english_summary(reg: RegressionResult, alpha: float = 0.05) -> list[str]:
    """Generates human-readable findings sentences from regression output."""
    findings = []
    for _, row in reg.coef_table.iterrows():
        term = row["term"]
        coef = row["coef"]
        p = row["p_value"]
        significance = "statistically significant" if p < alpha else "not statistically significant"
        direction = "increased" if coef > 0 else "decreased"
        findings.append(
            f"{term}: {direction} score by {abs(coef):.2f} points on average "
            f"(p={p:.3f}, {significance}), holding substance constant."
        )
    findings.append(
        f"Model explains {reg.r_squared*100:.1f}% of score variance across "
        f"{reg.n_obs} scored (resume, job description) pairs."
    )
    return findings


if __name__ == "__main__":
    df = load_merged_data()
    print(f"Loaded {len(df)} scored pairs across providers: {df['provider'].unique()}")

    for provider in df["provider"].unique():
        print(f"\n=== Provider: {provider} ===")
        sub = df[df["provider"] == provider]
        reg = run_regression(sub)
        print(reg.coef_table.to_string(index=False))
        print()
        for line in plain_english_summary(reg):
            print(f"  - {line}")
