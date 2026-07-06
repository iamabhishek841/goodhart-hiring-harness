"""
Streamlit dashboard for the Goodhart Stress-Test Harness.

Run locally:
    streamlit run dashboard/app.py

Deploy: push repo to GitHub, then connect it on share.streamlit.io
(Streamlit Community Cloud), pointing at dashboard/app.py as the entry file.
"""

import sys
import os
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis import (
    load_merged_data, run_regression, effect_size_deltas,
    plain_english_summary, PROXY_COLUMNS,
)
from prompts import SYSTEM_PROMPT, build_messages
from evaluate import PROVIDER_CONFIG, call_with_retry, parse_score_response
from openai import OpenAI

st.set_page_config(page_title="Goodhart Hiring Stress-Test", layout="wide")

VARIANTS_PATH = ROOT / "data" / "resume_variants.json"
RESULTS_PATH = ROOT / "results" / "scores.csv"
JD_PATH = ROOT / "data" / "job_descriptions.csv"


@st.cache_data
def load_variants():
    return json.loads(VARIANTS_PATH.read_text())


@st.cache_data
def load_jds():
    return pd.read_csv(JD_PATH)


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return None
    return pd.read_csv(RESULTS_PATH)


# ---------------------------------------------------------------------------
st.title("Goodhart Stress-Test Harness for LLM Hiring Rankers")
st.markdown(
    "**Research question:** when actual candidate competency is held constant, "
    "how much does an LLM hiring screener's score shift based on surface-level "
    "proxy signals alone -- university prestige, keyword density, employment "
    "gaps, and years-of-experience framing? This is a controlled study of "
    "Goodhart's Law failure modes in AI-driven hiring."
)

tab1, tab2, tab3 = st.tabs(["Live Demo", "Aggregate Findings", "Methodology & Limitations"])

# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Score two resume variants side by side")
    st.caption(
        "Pick two variants of the *same* base resume that differ only in a "
        "proxy feature, and a job description, then see how the live model "
        "scores them."
    )

    variants = load_variants()
    jds = load_jds()

    variant_ids = [v["variant_id"] for v in variants]
    variant_lookup = {v["variant_id"]: v for v in variants}

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        variant_a_id = st.selectbox("Variant A", variant_ids, index=0)
    with col_b:
        default_b = 1 if len(variant_ids) > 1 else 0
        variant_b_id = st.selectbox("Variant B", variant_ids, index=default_b)
    with col_c:
        jd_id = st.selectbox("Job description", jds["jd_id"].tolist())

    provider = st.radio("Provider", list(PROVIDER_CONFIG.keys()), horizontal=True)

    if st.button("Score both", type="primary"):
        config = PROVIDER_CONFIG[provider]
        api_key = os.environ.get(config["env_var"])
        if not api_key:
            st.error(
                f"Missing environment variable `{config['env_var']}`. "
                f"Set it (get a free key from Groq or Google AI Studio) and restart."
            )
        else:
            client = OpenAI(api_key=api_key, base_url=config["base_url"])
            jd_text = jds[jds["jd_id"] == jd_id]["description"].iloc[0]

            results = {}
            for label, vid in [("A", variant_a_id), ("B", variant_b_id)]:
                with st.spinner(f"Scoring variant {label}..."):
                    resume_text = variant_lookup[vid]["resume_text"]
                    messages = build_messages(jd_text, resume_text)
                    raw = call_with_retry(client, config["default_model"], messages)
                    parsed = parse_score_response(raw)
                    results[label] = parsed

            col_x, col_y = st.columns(2)
            with col_x:
                st.metric(f"Variant A ({variant_a_id})", f"{results['A']['score']}/10")
                st.caption(results["A"].get("justification", ""))
                with st.expander("Resume text"):
                    st.text(variant_lookup[variant_a_id]["resume_text"])
            with col_y:
                st.metric(f"Variant B ({variant_b_id})", f"{results['B']['score']}/10")
                st.caption(results["B"].get("justification", ""))
                with st.expander("Resume text"):
                    st.text(variant_lookup[variant_b_id]["resume_text"])

            delta = results["A"]["score"] - results["B"]["score"]
            if delta != 0:
                st.warning(
                    f"Score delta: {delta:+d} points, despite identical underlying "
                    f"substance between these two variants."
                )
            else:
                st.success("No score delta between these two variants.")

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Aggregate findings across all scored pairs")
    results_df = load_results()

    if results_df is None or results_df.empty:
        st.info(
            "No results yet. Run `python src/evaluate.py --provider groq` "
            "(or `--provider gemini`) from the project root to populate "
            "`results/scores.csv`, then reload this dashboard."
        )
    else:
        providers_available = results_df["provider"].unique().tolist()
        selected_provider = st.selectbox("Provider", providers_available)

        merged = load_merged_data(provider=selected_provider)

        st.markdown("#### Mean score by proxy feature level")
        deltas = effect_size_deltas(merged)
        fig = go.Figure()
        for feature in deltas["proxy_feature"].unique():
            sub = deltas[deltas["proxy_feature"] == feature]
            fig.add_trace(go.Bar(
                x=[f"{feature}: {lvl}" for lvl in sub["level"]],
                y=sub["mean_score"],
                error_y=dict(type="data", array=sub["std_score"]),
                name=feature,
            ))
        fig.update_layout(
            yaxis_title="Mean LLM score (1-10)",
            xaxis_title="Proxy feature level",
            showlegend=False,
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Regression coefficients (proxy effect on score, substance held constant)")
        reg = run_regression(merged)
        coef_df = reg.coef_table
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=coef_df["coef"], y=coef_df["term"],
            error_x=dict(
                type="data",
                symmetric=False,
                array=coef_df["ci_upper"] - coef_df["coef"],
                arrayminus=coef_df["coef"] - coef_df["ci_lower"],
            ),
            mode="markers", marker=dict(size=10),
        ))
        fig2.add_vline(x=0, line_dash="dash", line_color="gray")
        fig2.update_layout(
            xaxis_title="Effect on score (points)", yaxis_title="",
            height=350 + 20 * len(coef_df),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Plain-English summary")
        for line in plain_english_summary(reg):
            st.markdown(f"- {line}")

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Methodology & Limitations")
    st.markdown("""
**Design.** Each base resume's substance block (real project descriptions,
demonstrated skills) is held byte-identical across every variant. Only proxy
signals are toggled: university tier (top-tier / unranked / omitted),
keyword framing (stuffed / natural), employment gap (present / absent), and
years-of-experience framing (explicit / project-only). Any score difference
between variants of the same base resume is therefore attributable to the
toggled proxy feature, not to a real competency difference.

**Sample size.** This is a small controlled study (a handful of base resumes,
a partial-factorial sample of proxy combinations, 2-3 job descriptions), not
a large-scale audit. Effect sizes here should be read as a demonstration of
mechanism, not as a precise estimate of real-world bias magnitude at
population scale.

**Model scope.** Results are shown per-provider (Groq's Llama 3.3 70B,
Google's Gemini Flash). These are useful, capable models but are not
identical to the proprietary systems many real hiring platforms use
internally, and results may not generalize to those systems.

**Prompt sensitivity.** LLM outputs are sensitive to prompt phrasing. A
different scoring prompt, temperature, or output format could shift results.
The prompt used here (see `src/prompts.py`) was fixed across all calls to
control for this within this study, but was not itself validated against
multiple phrasings.

**What this study does NOT claim.** It does not claim that all AI hiring
tools exhibit this exact bias, nor does it quantify real-world hiring
outcomes. It demonstrates a specific, reproducible mechanism -- proxy
sensitivity under constant substance -- using open, inspectable methodology.
""")
