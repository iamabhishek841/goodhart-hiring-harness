"""
Research-facing Streamlit dashboard for the Goodhart Stress-Test Harness.

Run locally:
    streamlit run dashboard/app.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis import (  # noqa: E402
    effect_size_deltas,
    load_merged_data,
    run_regression,
)
from evaluate import PROVIDER_CONFIG, call_with_retry, parse_score_response  # noqa: E402
from prompts import build_messages  # noqa: E402

st.set_page_config(
    page_title="Goodhart Hiring Harness",
    layout="wide",
    initial_sidebar_state="collapsed",
)

VARIANTS_PATH = ROOT / "data" / "resume_variants.json"
RESULTS_PATH = ROOT / "results" / "scores.csv"
JD_PATH = ROOT / "data" / "job_descriptions.csv"
EXPECTED_PAIRS_PER_PROVIDER = 288

PROXY_LABELS = {
    "university_tier": "University signal",
    "keyword_framing": "Keyword framing",
    "employment_gap": "Employment gap",
    "years_framing": "Experience framing",
}

PROVIDER_LABELS = {
    "groq": "Groq / Llama 3.3 70B",
    "gemini": "Gemini 2.5 Flash",
}

LEVEL_LABELS = {
    "top_tier": "Top-tier university",
    "unranked": "Unranked university",
    "omitted": "Education omitted",
    "stuffed": "Keyword-stuffed",
    "natural": "Natural language",
    "gap_present": "Gap disclosed",
    "no_gap": "No gap",
    "explicit_years": "Explicit years",
    "project_only": "Project-only evidence",
}

ACCENT = "#26547C"
TEAL = "#2A9D8F"
AMBER = "#E9A227"
RED = "#C44536"
INK = "#18212F"
MUTED = "#667085"

st.markdown(
    f"""
    <style>
    .main .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }}
    .hero {{
        border: 1px solid #D9E2EC;
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        background: #FFFFFF;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        margin-bottom: 0.75rem;
    }}
    .eyebrow {{
        color: {ACCENT};
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }}
    .hero h1 {{
        color: {INK};
        font-size: clamp(2.0rem, 4vw, 3.15rem);
        line-height: 1.05;
        margin: 0 0 0.45rem 0;
        letter-spacing: 0;
    }}
    .hero p {{
        color: #344054;
        max-width: 920px;
        font-size: 1.02rem;
        line-height: 1.52;
        margin-bottom: 0.15rem;
    }}
    .finding {{
        border-left: 5px solid {TEAL};
        background: #F3FAF8;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        color: #1E443F;
        margin: 0.6rem 0 1rem 0;
    }}
    .warning-note {{
        border-left: 5px solid {AMBER};
        background: #FFFAEB;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        color: #5F3B00;
        margin: 0.6rem 0 1rem 0;
    }}
    .mini-card {{
        border: 1px solid #E4E7EC;
        border-radius: 8px;
        background: #FFFFFF;
        padding: 0.95rem 1rem;
        min-height: 125px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
    }}
    .mini-card h4 {{
        color: {INK};
        font-size: 1.0rem;
        margin: 0 0 0.35rem 0;
    }}
    .mini-card p {{
        color: {MUTED};
        font-size: 0.93rem;
        line-height: 1.48;
        margin: 0;
    }}
    .method-band {{
        border: 1px solid #E4E7EC;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #F8FAFC;
        margin-bottom: 0.8rem;
    }}
    .method-band strong {{
        color: {INK};
    }}
    .small-muted {{
        color: {MUTED};
        font-size: 0.88rem;
    }}
    [data-testid="stMetricValue"] {{
        color: {INK};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_variants() -> list[dict]:
    return json.loads(VARIANTS_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_jds() -> pd.DataFrame:
    return pd.read_csv(JD_PATH)


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS_PATH)


def get_api_key(env_var: str, user_supplied_key: str | None = None) -> str | None:
    if user_supplied_key and user_supplied_key.strip():
        return user_supplied_key.strip()
    try:
        secret = st.secrets.get(env_var, None)
    except Exception:
        secret = None
    return secret or os.environ.get(env_var)


def provider_counts(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(columns=["provider", "scored_pairs", "completion"])
    counts = results_df.groupby("provider").size().reset_index(name="scored_pairs")
    counts["completion"] = (counts["scored_pairs"] / EXPECTED_PAIRS_PER_PROVIDER).clip(upper=1.0)
    counts["completion_label"] = (counts["completion"] * 100).round(1).astype(str) + "%"
    return counts.sort_values(["scored_pairs", "provider"], ascending=[False, True])


def pretty_level(value: str) -> str:
    return LEVEL_LABELS.get(str(value), str(value).replace("_", " ").title())


def pretty_feature(value: str) -> str:
    return PROXY_LABELS.get(str(value), str(value).replace("_", " ").title())


def pretty_provider(value: str) -> str:
    return PROVIDER_LABELS.get(str(value), str(value).title())


def pretty_term(term: str) -> str:
    for feature, label in PROXY_LABELS.items():
        marker = f"C({feature})[T."
        if marker in term:
            level = term.split("[T.", 1)[1].rstrip("]")
            return f"{label}: {pretty_level(level)}"
    return term


def format_number(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def format_p_value(value: float) -> str:
    if value < 0.001:
        return "p < 0.001"
    return f"p = {value:.3f}"


def regression_for_provider(provider: str):
    merged = load_merged_data(provider=provider)
    return merged, run_regression(merged)


def best_available_provider(counts_df: pd.DataFrame) -> str | None:
    if counts_df.empty:
        return None
    return str(counts_df.iloc[0]["provider"])


def strongest_effect(reg) -> pd.Series | None:
    coef_df = reg.coef_table.copy()
    if coef_df.empty:
        return None
    coef_df["abs_coef"] = coef_df["coef"].abs()
    return coef_df.sort_values("abs_coef", ascending=False).iloc[0]


def significant_effects(reg, alpha: float = 0.05) -> pd.DataFrame:
    coef_df = reg.coef_table.copy()
    if coef_df.empty:
        return coef_df
    return coef_df[coef_df["p_value"] < alpha].copy()


def readable_summary(reg, alpha: float = 0.05) -> list[str]:
    findings = []
    for _, row in reg.coef_table.iterrows():
        significance = "statistically significant" if row["p_value"] < alpha else "not statistically significant"
        direction = "increased" if row["coef"] > 0 else "decreased"
        findings.append(
            f"{pretty_term(row['term'])}: {direction} score by {abs(row['coef']):.2f} points on average "
            f"(p={row['p_value']:.3f}, {significance}), holding substance constant."
        )
    findings.append(
        f"Model explains {reg.r_squared * 100:.1f}% of score variance across "
        f"{reg.n_obs} scored resume/job-description pairs."
    )
    return findings


def executive_findings(reg) -> list[str]:
    top = strongest_effect(reg)
    if top is None:
        return ["No proxy-effect terms were available for this provider slice yet."]

    significant = significant_effects(reg)
    findings = [
        (
            f"The strongest observed proxy effect is {pretty_term(top['term'])}, "
            f"which shifts the score by {top['coef']:+.2f} points "
            f"({format_p_value(top['p_value'])}) while candidate substance is held constant."
        )
    ]

    if significant.empty:
        findings.append(
            "No proxy effect is statistically significant yet, so this provider slice should be treated as preliminary."
        )
    else:
        sig_labels = ", ".join(pretty_term(term) for term in significant["term"])
        findings.append(
            f"Statistically significant proxy sensitivity appears for: {sig_labels}."
        )

    findings.append(
        f"The regression explains {reg.r_squared * 100:.1f}% of score variance across "
        f"{reg.n_obs} scored resume/job-description pairs."
    )
    findings.append(
        "Other proxy effects in this run are small and should not be over-interpreted without a larger completed run."
    )
    return findings


def make_regression_plot(coef_df: pd.DataFrame) -> go.Figure:
    plot_df = coef_df.copy()
    plot_df["label"] = plot_df["term"].map(pretty_term)
    plot_df = plot_df.sort_values("coef")
    colors = [TEAL if p < 0.05 else "#98A2B3" for p in plot_df["p_value"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["coef"],
            y=plot_df["label"],
            mode="markers",
            marker=dict(size=12, color=colors, line=dict(width=1, color="#344054")),
            error_x=dict(
                type="data",
                symmetric=False,
                array=plot_df["ci_upper"] - plot_df["coef"],
                arrayminus=plot_df["coef"] - plot_df["ci_lower"],
                color="#98A2B3",
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Effect: %{x:.2f} score points<br>"
                "95% CI shown by whiskers<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#475467")
    fig.update_layout(
        xaxis_title="Estimated effect on score, holding resume substance constant",
        yaxis_title="",
        height=max(390, 310 + 34 * len(plot_df)),
        margin=dict(l=8, r=18, t=12, b=24),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=INK),
    )
    fig.update_xaxes(gridcolor="#EEF2F6", zeroline=False)
    fig.update_yaxes(gridcolor="#F7F9FC")
    return fig


def make_feature_means_plot(deltas: pd.DataFrame) -> go.Figure:
    plot_df = deltas.copy()
    plot_df["feature_label"] = plot_df["proxy_feature"].map(pretty_feature)
    plot_df["level_label"] = plot_df["level"].map(pretty_level)

    palette = [ACCENT, TEAL, AMBER, RED]
    fig = go.Figure()
    for idx, feature in enumerate(plot_df["proxy_feature"].drop_duplicates()):
        sub = plot_df[plot_df["proxy_feature"] == feature]
        fig.add_trace(
            go.Bar(
                x=[f"{pretty_feature(feature)}<br>{level}" for level in sub["level_label"]],
                y=sub["mean_score"],
                error_y=dict(type="data", array=sub["std_score"], color="#98A2B3"),
                marker_color=palette[idx % len(palette)],
                name=pretty_feature(feature),
                hovertemplate="Mean score: %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(
        yaxis_title="Mean LLM score (1-10)",
        xaxis_title="Proxy feature level",
        showlegend=False,
        height=460,
        margin=dict(l=8, r=18, t=12, b=88),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=INK),
    )
    fig.update_yaxes(gridcolor="#EEF2F6", range=[0, 10])
    fig.update_xaxes(tickfont=dict(size=11))
    return fig


def make_markdown_report(provider: str, merged: pd.DataFrame, reg) -> str:
    lines = [
        "# Goodhart Stress-Test Harness Report",
        "",
        f"Provider: **{pretty_provider(provider)}**",
        f"Scored resume/job pairs: **{len(merged)}**",
        f"Regression R-squared: **{reg.r_squared:.3f}**",
        "",
        "## Research question",
        (
            "When candidate substance is held constant, do LLM hiring rankers shift "
            "scores based on surface-level proxy signals such as university prestige, "
            "keyword density, employment gaps, and years-of-experience framing?"
        ),
        "",
        "## Main findings",
    ]
    for item in readable_summary(reg):
        lines.append(f"- {item}")
    lines += [
        "",
        "## Method note",
        (
            "Each resume variant preserves the same underlying competency block while "
            "toggling only proxy signals. Score shifts are therefore interpreted as "
            "proxy sensitivity, not differences in candidate substance."
        ),
        "",
        "## Limits",
        (
            "This is a controlled stress test, not a population-scale audit or a claim "
            "about every production hiring system."
        ),
    ]
    return "\n".join(lines)


def find_controlled_pair(
    variants: list[dict],
    base_resume_id: str,
    feature: str,
    first_level: str,
    second_level: str,
) -> tuple[dict | None, dict | None]:
    other_features = [name for name in PROXY_LABELS if name != feature]
    candidates = [v for v in variants if v["base_resume_id"] == base_resume_id]

    for left in candidates:
        if left[feature] != first_level:
            continue
        for right in candidates:
            if right[feature] != second_level:
                continue
            if all(left[name] == right[name] for name in other_features):
                return left, right
    return None, None


results_df = load_results()
counts_df = provider_counts(results_df)
default_provider = best_available_provider(counts_df)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">AI safety evaluation / Goodhart's Law / hiring rankers</div>
      <h1>Goodhart Stress-Test Harness for LLM Hiring Rankers</h1>
      <p>
        A controlled audit that asks whether an LLM hiring screener rewards surface-level proxy
        signals even when the candidate's underlying project substance is held fixed.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if default_provider:
    try:
        overview_merged, overview_reg = regression_for_provider(default_provider)
        top_effect = strongest_effect(overview_reg)
        sig = significant_effects(overview_reg)

        if top_effect is not None:
            effect_text = (
                f"On {pretty_provider(default_provider)}, the largest observed proxy shift is "
                f"{pretty_term(top_effect['term'])}: {top_effect['coef']:+.2f} score points "
                f"(p={top_effect['p_value']:.3g}) across {len(overview_merged)} scored pairs."
            )
        else:
            effect_text = "Regression completed, but no proxy-effect terms were available."

        st.markdown(f"<div class='finding'><strong>Main evidence:</strong> {effect_text}</div>", unsafe_allow_html=True)
        if sig.empty:
            st.markdown(
                "<div class='warning-note'><strong>Interpretation:</strong> No proxy term is statistically significant yet for the current provider slice. Treat results as preliminary until more scoring is complete.</div>",
                unsafe_allow_html=True,
            )
    except Exception as exc:
        st.warning(f"Could not run overview regression yet: {exc}")
else:
    st.info("No scored results found yet. Run the evaluator to populate results/scores.csv.")

tab_overview, tab_evidence, tab_method, tab_live = st.tabs(
    ["Executive Overview", "Evidence", "Methodology", "Optional Live Test"]
)


with tab_overview:
    st.subheader("Executive overview")

    if results_df.empty:
        st.warning("No results are available yet. The dashboard will populate after an evaluation run.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Scored pairs", len(results_df))
        with c2:
            st.metric("Providers", results_df["provider"].nunique())
        with c3:
            st.metric("Resume variants", results_df["variant_id"].nunique())
        with c4:
            st.metric("Job descriptions", results_df["jd_id"].nunique())

        st.markdown("#### Run status")
        display_counts = counts_df[["provider", "scored_pairs", "completion_label"]].rename(
            columns={
                "provider": "Provider",
                "scored_pairs": "Scored pairs",
                "completion_label": "Completion vs 288/provider",
            }
        )
        st.dataframe(display_counts, width="stretch", hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="mini-card">
              <h4>Safety question</h4>
              <p>Can a model be pushed toward proxy optimization when asked to make a high-impact screening decision?</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="mini-card">
              <h4>Controlled design</h4>
              <p>The underlying project substance is fixed; only presentation-level proxy signals are toggled.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="mini-card">
              <h4>Research output</h4>
              <p>The result is an auditable workflow: generated variants, fixed prompt, logged scores, regression, and limitations.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if default_provider:
        try:
            merged, reg = regression_for_provider(default_provider)
            st.markdown("#### Key findings")
            for line in executive_findings(reg):
                st.markdown(f"- {line}")

            report = make_markdown_report(default_provider, merged, reg)
            st.download_button(
                "Download audit summary",
                data=report,
                file_name=f"goodhart_hiring_harness_{default_provider}_summary.md",
                mime="text/markdown",
            )
        except Exception as exc:
            st.error(f"Could not summarize results yet: {exc}")


with tab_evidence:
    st.subheader("Evidence and regression")

    if results_df.empty:
        st.info("Run `python src/evaluate.py --provider groq` to populate `results/scores.csv`.")
    else:
        provider_options = counts_df["provider"].tolist()
        selected_provider = st.selectbox("Provider", provider_options, index=0)
        merged = load_merged_data(provider=selected_provider)

        if len(merged) < 50:
            st.warning(
                f"{selected_provider} has only {len(merged)} scored pairs. "
                "Use this for smoke-test inspection, not final inference."
            )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Analysed pairs", len(merged))
        with m2:
            st.metric("Mean score", format_number(merged["score"].mean()))
        with m3:
            st.metric(
                "Score range",
                f"{format_number(merged['score'].min(), 1)} to {format_number(merged['score'].max(), 1)}",
            )
        with m4:
            completion = min(len(merged) / EXPECTED_PAIRS_PER_PROVIDER, 1.0)
            st.metric("Provider completion", f"{completion * 100:.1f}%")

        try:
            reg = run_regression(merged)
            coef_df = reg.coef_table.copy()
            coef_df["Readable term"] = coef_df["term"].map(pretty_term)

            st.markdown("#### Proxy effect on score")
            st.caption(
                "Dots show regression coefficients. Positive values mean the proxy level increased the screener score after controlling for base resume substance."
            )
            st.plotly_chart(make_regression_plot(coef_df), width="stretch")

            with st.expander("Coefficient table"):
                table = coef_df[
                    ["Readable term", "coef", "ci_lower", "ci_upper", "p_value"]
                ].rename(
                    columns={
                        "coef": "Effect",
                        "ci_lower": "95% CI lower",
                        "ci_upper": "95% CI upper",
                        "p_value": "p-value",
                    }
                )
                st.dataframe(table, width="stretch", hide_index=True)

            st.markdown("#### Mean score by proxy feature level")
            deltas = effect_size_deltas(merged)
            st.plotly_chart(make_feature_means_plot(deltas), width="stretch")

            st.markdown("#### Interpretation guardrails")
            st.markdown(
                """
                - A score shift under constant substance is evidence of proxy sensitivity.
                - This is a controlled mechanism test, not a population-scale discrimination estimate.
                - Completed provider runs are more reliable than smoke-test slices.
                """
            )
        except Exception as exc:
            st.error(f"Regression failed for this provider: {exc}")


with tab_method:
    st.subheader("Methodology")
    st.markdown(
        """
        <div class="method-band">
        <strong>Core design:</strong> keep demonstrated candidate substance byte-identical within each base resume,
        then toggle only proxy signals. If the score changes, the model is reacting to presentation-level proxies,
        not new evidence of candidate capability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Held constant")
        st.markdown(
            """
            - Base resume project substance
            - Job description within each comparison
            - Scoring prompt and JSON output format
            - Temperature and provider configuration
            """
        )
    with right:
        st.markdown("#### Toggled proxy signals")
        st.markdown(
            """
            - University prestige signal
            - Keyword-stuffed vs natural framing
            - Employment gap disclosure
            - Explicit years vs project-only experience framing
            """
        )

    st.markdown("#### Reproducibility pipeline")
    st.code(
        """python src/generate_variants.py
pytest tests/
python src/evaluate.py --provider groq
python src/analysis.py
streamlit run dashboard/app.py""",
        language="bash",
    )

    st.markdown("#### Why this is AI safety work")
    st.markdown(
        """
        This project treats hiring as a high-impact AI system where the model can optimize for proxies that are easy to measure,
        imitate, or overfit. The harness translates a responsible-AI concern into an inspectable evaluation: controlled inputs,
        fixed prompts, logged outputs, statistical analysis, and explicit limits on the claim.
        """
    )

    st.markdown("#### Limitations")
    st.markdown(
        """
        - The experiment demonstrates a mechanism; it is not a population-scale bias audit.
        - Results are provider- and prompt-specific.
        - Real hiring platforms may use additional ranking, retrieval, or human-review layers.
        - The study should be extended with more job descriptions, prompt variants, and model providers before making broad claims.
        """
    )


with tab_live:
    st.subheader("Optional live stress test")
    st.caption(
        "Use this for a quick demo. The aggregate run in the Evidence tab is the stronger result."
    )

    missing_files = [p for p in [VARIANTS_PATH, JD_PATH] if not p.exists()]
    if missing_files:
        st.error("Missing required files: " + ", ".join(str(path) for path in missing_files))
    else:
        variants = load_variants()
        jds = load_jds()
        base_ids = sorted({v["base_resume_id"] for v in variants})

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            base_id = st.selectbox("Base resume", base_ids)
        with col_b:
            feature = st.selectbox(
                "Proxy to toggle",
                list(PROXY_LABELS.keys()),
                format_func=pretty_feature,
                index=1,
            )
        with col_c:
            jd_id = st.selectbox("Job description", jds["jd_id"].tolist())

        levels = sorted({v[feature] for v in variants if v["base_resume_id"] == base_id})
        left_level, right_level = st.columns(2)
        with left_level:
            first_level = st.selectbox("Variant A level", levels, format_func=pretty_level, index=0)
        with right_level:
            default_second = 1 if len(levels) > 1 else 0
            second_level = st.selectbox(
                "Variant B level",
                levels,
                format_func=pretty_level,
                index=default_second,
            )

        variant_a, variant_b = find_controlled_pair(
            variants, base_id, feature, first_level, second_level
        )

        if variant_a is None or variant_b is None:
            st.warning("No controlled pair found for this exact selection.")
        else:
            st.markdown(
                f"Comparing `{variant_a['variant_id']}` and `{variant_b['variant_id']}`. "
                f"They differ on `{feature}` while the other proxy labels match."
            )

            with st.expander("Preview resume variants"):
                left_text, right_text = st.columns(2)
                with left_text:
                    st.markdown(f"**Variant A: {variant_a['variant_id']}**")
                    st.text(variant_a["resume_text"])
                with right_text:
                    st.markdown(f"**Variant B: {variant_b['variant_id']}**")
                    st.text(variant_b["resume_text"])

            st.markdown("#### API key for live testing")
            st.caption(
                "Visitors can use their own keys for live calls. The dashboard does not write these keys to disk."
            )
            key_col_1, key_col_2 = st.columns(2)
            with key_col_1:
                groq_key = st.text_input(
                    "Groq API key",
                    type="password",
                    placeholder="gsk_...",
                    help="Used only when Provider is set to groq.",
                )
            with key_col_2:
                google_key = st.text_input(
                    "Google AI Studio API key",
                    type="password",
                    placeholder="AIza...",
                    help="Used only when Provider is set to gemini.",
                )

            provider = st.radio("Provider", list(PROVIDER_CONFIG.keys()), horizontal=True)
            manual_keys = {
                "groq": groq_key,
                "gemini": google_key,
            }

            if st.button("Score controlled pair", type="primary"):
                config = PROVIDER_CONFIG[provider]
                api_key = get_api_key(config["env_var"], manual_keys.get(provider))
                if not api_key:
                    st.error(
                        f"Missing `{config['env_var']}`. Paste a key above, or add it to Streamlit secrets/environment."
                    )
                else:
                    client = OpenAI(api_key=api_key, base_url=config["base_url"])
                    jd_text = jds[jds["jd_id"] == jd_id]["description"].iloc[0]

                    live_results = {}
                    for label, variant in [("A", variant_a), ("B", variant_b)]:
                        with st.spinner(f"Scoring variant {label} with {provider}..."):
                            messages = build_messages(jd_text, variant["resume_text"])
                            raw = call_with_retry(client, config["default_model"], messages)
                            live_results[label] = parse_score_response(raw)

                    score_a = float(live_results["A"]["score"])
                    score_b = float(live_results["B"]["score"])
                    delta = score_a - score_b

                    out_a, out_b, out_delta = st.columns(3)
                    with out_a:
                        st.metric(f"Variant A ({variant_a['variant_id']})", f"{score_a:.0f}/10")
                        st.caption(live_results["A"].get("justification", ""))
                    with out_b:
                        st.metric(f"Variant B ({variant_b['variant_id']})", f"{score_b:.0f}/10")
                        st.caption(live_results["B"].get("justification", ""))
                    with out_delta:
                        st.metric("Score delta", f"{delta:+.1f}")

                    if delta:
                        st.warning(
                            "The model produced a score difference for a controlled pair. "
                            "Treat this as a live illustration; aggregate regression is stronger evidence."
                        )
                    else:
                        st.success("No score difference in this live comparison.")
