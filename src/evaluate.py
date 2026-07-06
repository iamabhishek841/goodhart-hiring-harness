"""
evaluate.py

Runs every (resume_variant, job_description) pair through an LLM hiring-
screener prompt and logs the score. Supports Groq and Gemini via their
OpenAI-compatible endpoints, both free, no credit card required.

Usage:
    export GROQ_API_KEY=...          # or GOOGLE_API_KEY for gemini
    python evaluate.py --provider groq --limit 20      # cheap smoke test
    python evaluate.py --provider groq                 # full run
    python evaluate.py --provider gemini                # run on Gemini too, for cross-model comparison

Checkpointing: already-scored (variant_id, jd_id, provider) combos are
skipped on re-run, so an interrupted run can be safely resumed.
"""

import argparse
import json
import os
import sys
import time
import csv
from pathlib import Path
from datetime import datetime, timezone

from openai import OpenAI
import pandas as pd

from prompts import SYSTEM_PROMPT, build_messages

ROOT = Path(__file__).resolve().parent.parent
VARIANTS_PATH = ROOT / "data" / "resume_variants.json"
JD_PATH = ROOT / "data" / "job_descriptions.csv"
RESULTS_PATH = ROOT / "results" / "scores.csv"
DEBUG_LOG_PATH = ROOT / "results" / "debug_log.jsonl"

PROVIDER_CONFIG = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_var": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "default_sleep": 2.0,   # Groq's free tier allows ~30 RPM
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_var": "GOOGLE_API_KEY",
        "default_model": "gemini-2.5-flash",
        "default_sleep": 13.0,  # Free tier observed limit: 5 requests/minute -> ~12s minimum gap + buffer
    },
}

RESULTS_FIELDS = [
    "variant_id",
    "jd_id",
    "provider",
    "model",
    "score",
    "justification",
    "timestamp",
]


def load_existing_keys() -> set[tuple[str, str, str]]:
    """Returns set of (variant_id, jd_id, provider) already scored."""
    if not RESULTS_PATH.exists():
        return set()
    df = pd.read_csv(RESULTS_PATH)
    return set(zip(df["variant_id"], df["jd_id"], df["provider"]))


def append_result(row: dict):
    file_exists = RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_debug(entry: dict):
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def parse_score_response(raw_text: str) -> dict:
    """Best-effort JSON parse; falls back to regex if the model adds stray text."""
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        import re

        score_match = re.search(r'"score"\s*:\s*(\d+)', raw_text)
        just_match = re.search(r'"justification"\s*:\s*"([^"]*)"', raw_text)
        if score_match:
            return {
                "score": int(score_match.group(1)),
                "justification": just_match.group(1) if just_match else "",
            }
        raise ValueError(f"Could not parse model response: {raw_text[:200]}")


import re as _re


def parse_retry_after_seconds(error_message: str) -> float | None:
    """
    Extracts a suggested wait duration from a provider error message.
    Groq phrases this as 'try again in 5m30.048s'; Gemini phrases it as
    'retry in 41.21s'. Match both.
    """
    match = _re.search(r"(?:try again|retry) in (?:(\d+)m)?(\d+(?:\.\d+)?)s", error_message)
    if not match:
        return None
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def call_with_retry(client: OpenAI, model: str, messages: list, max_retries: int = 4,
                     max_quota_waits: int = 4):
    """
    Two kinds of failures are handled differently:
    - Transient errors (network blip, momentary overload): short exponential
      backoff, counted against max_retries.
    - Daily/quota rate limits ("tokens per day" exceeded): these won't
      resolve in a few seconds. We parse the provider's suggested wait time
      and sleep for that long instead of giving up, up to max_quota_waits
      times, so a full run can self-heal without manual restarts.
    """
    quota_waits = 0
    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                temperature=0,
                max_tokens=800,
            )
            return response.choices[0].message.content
        except Exception as e:
            msg = str(e)
            retry_after = parse_retry_after_seconds(msg)

            if retry_after is not None and quota_waits < max_quota_waits:
                quota_waits += 1
                wait = retry_after + 5  # small buffer
                print(f"  [quota wait {quota_waits}/{max_quota_waits}] daily/rate limit hit -- "
                      f"sleeping {wait:.0f}s as suggested by provider")
                time.sleep(wait)
                continue

            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Failed after {max_retries} retries and {quota_waits} quota waits: {msg}")
            wait = 2**attempt
            print(f"  [retry {attempt}/{max_retries}] error: {msg} -- waiting {wait}s")
            time.sleep(wait)


def run(provider: str, model: str | None, limit: int | None, sleep_seconds: float | None):
    config = PROVIDER_CONFIG[provider]
    api_key = os.environ.get(config["env_var"])
    if not api_key:
        print(f"ERROR: environment variable {config['env_var']} is not set.")
        sys.exit(1)

    model = model or config["default_model"]
    sleep_seconds = sleep_seconds if sleep_seconds is not None else config["default_sleep"]
    client = OpenAI(api_key=api_key, base_url=config["base_url"])

    variants = json.loads(VARIANTS_PATH.read_text())
    jds = pd.read_csv(JD_PATH).to_dict("records")

    existing = load_existing_keys()
    pairs = [(v, jd) for v in variants for jd in jds]
    if limit:
        pairs = pairs[:limit]

    print(f"Provider={provider} model={model} | {len(pairs)} pairs queued "
          f"({len(existing)} already scored, will skip those)")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    for variant, jd in pairs:
        key = (variant["variant_id"], jd["jd_id"], provider)
        if key in existing:
            continue

        messages = build_messages(jd["description"], variant["resume_text"])
        raw = call_with_retry(client, model, messages)
        log_debug({"variant_id": variant["variant_id"], "jd_id": jd["jd_id"],
                   "provider": provider, "raw_response": raw})

        try:
            parsed = parse_score_response(raw)
        except ValueError as e:
            print(f"  SKIPPING unparseable response: {e}")
            continue

        append_result({
            "variant_id": variant["variant_id"],
            "jd_id": jd["jd_id"],
            "provider": provider,
            "model": model,
            "score": parsed["score"],
            "justification": parsed.get("justification", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        done += 1
        print(f"  [{done}/{len(pairs)}] {variant['variant_id']} x {jd['jd_id']} -> score={parsed['score']}")
        time.sleep(sleep_seconds)

    print(f"Done. {done} new scores written to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=list(PROVIDER_CONFIG.keys()), default="groq")
    parser.add_argument("--model", default=None, help="Override default model for the provider")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N pairs (cheap smoke test)")
    parser.add_argument("--sleep", type=float, default=None,
                         help="Seconds to sleep between calls. Defaults to a provider-safe value if omitted.")
    args = parser.parse_args()

    run(args.provider, args.model, args.limit, args.sleep)
