# Goodhart Stress-Test Harness for LLM Hiring Rankers

**Research question:** when actual candidate competency is held constant, how much
does an LLM hiring screener's score shift based on surface-level proxy signals
alone -- university prestige, keyword density, employment gaps, and
years-of-experience framing? This project is a controlled empirical study of
Goodhart's Law failure modes in AI-driven hiring systems, built as an applied
AI safety / responsible AI evaluation exercise.

Unlike a rule-based "risk checklist" tool, this project runs an actual
experiment: identical underlying substance (real project descriptions, real
demonstrated skill) is held byte-identical across resume variants, while only
proxy features are toggled. Any score difference between variants is
therefore attributable to proxy bias, not competency difference, and is
tested statistically via regression.

## Why this design

A base resume's "substance block" (concrete project work, demonstrated
skills) never changes across its variants. Four proxy dimensions are toggled
independently in a full-factorial design:

| Proxy dimension | Levels |
|---|---|
| University tier | top-tier / unranked / omitted |
| Keyword framing | keyword-stuffed / natural language |
| Employment gap | present / absent |
| Years-of-experience framing | explicit years stated / project-only |

4 base resumes x 24 combinations = 96 variants, scored against 3 job
descriptions = 288 scoring calls per model provider.

## Setup

```bash
pip install -r requirements.txt
```

Get a free API key (no credit card required) from at least one of:
- **Groq** -- console.groq.com -- fast, Llama 3.3 70B
- **Google AI Studio (Gemini)** -- aistudio.google.com -- Gemini 2.5 Flash

Set the relevant environment variable:

```bash
export GROQ_API_KEY="your-key-here"
# and/or
export GOOGLE_API_KEY="your-key-here"
```

## Running the pipeline

```bash
# 1. Generate controlled resume variants (no API calls, instant)
python src/generate_variants.py

# 2. Run tests to confirm substance-identity holds
pytest tests/

# 3. Smoke-test the evaluation call (cheap, ~5 calls)
python src/evaluate.py --provider groq --limit 5

# 4. Full evaluation run (288 calls, a few minutes with rate limiting)
python src/evaluate.py --provider groq

# Optional: run on Gemini too, for cross-model comparison
python src/evaluate.py --provider gemini

# 5. View regression analysis in the terminal
python src/analysis.py
```

## Running the dashboard locally

```bash
streamlit run dashboard/app.py
```

The dashboard includes an optional live stress test. Visitors can paste their
own Groq or Google AI Studio API key into password fields in the app, so the
demo does not depend on the maintainer's API quota.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app," select this repo, and set the main file path to
   `dashboard/app.py`.
4. Under "Secrets," add `GROQ_API_KEY` and/or `GOOGLE_API_KEY` so the live
   demo tab works for visitors.
5. Deploy. The Aggregate Findings tab will read from `results/scores.csv`,
   which you should commit to the repo after running the evaluation locally
   so the dashboard has data to show without visitors needing their own key.

## Repository structure

```
goodhart-hiring-harness/
├── data/
│   ├── job_descriptions.csv
│   ├── base_resumes/           # 4 hand-authored base resumes
│   └── resume_variants.json    # generated, 96 controlled variants
├── src/
│   ├── generate_variants.py
│   ├── prompts.py
│   ├── evaluate.py
│   └── analysis.py
├── dashboard/
│   └── app.py                  # Streamlit app: Live Demo / Findings / Methodology
├── results/
│   └── scores.csv               # populated by evaluate.py
├── tests/
│   └── test_generate_variants.py
└── requirements.txt
```

## Findings

Current results include 214 scored resume/job-description pairs across Groq
and Gemini. The Groq/Llama run is the main interpretable slice so far, with
202 scored pairs.

For Groq's Llama 3.3 70B, keyword-stuffed resume framing increased scores by
0.58 points on average (p < 0.001), even though the underlying candidate
substance was held constant. University framing, employment-gap disclosure,
and years-of-experience framing did not show statistically significant effects
in this run.

This supports the central safety concern of the project: an LLM screener can
reward presentation-level proxy optimization rather than only demonstrated
candidate capability.

Gemini currently has only 12 scored pairs in `results/scores.csv`, so its
results should be treated as a smoke test until the full provider run is
complete.

## Limitations

- Small controlled study (4 base resumes, partial job-description set) --
  demonstrates a mechanism, not a population-scale bias estimate.
- Results shown are specific to the model(s) evaluated (Groq's Llama 3.3 70B
  and/or Gemini 2.5 Flash), not necessarily generalizable to proprietary
  hiring systems.
- LLM outputs are sensitive to prompt phrasing; the scoring prompt was fixed
  across all calls to control for this within this study but was not itself
  validated against multiple phrasings.
- This project does not claim all AI hiring tools exhibit this exact bias --
  it demonstrates a specific, reproducible proxy-sensitivity mechanism under
  controlled conditions.
