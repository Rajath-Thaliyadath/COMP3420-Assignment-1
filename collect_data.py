"""
Data collection script for COMP8420 NLP assignment.
Fetches 200 human (Wikipedia) summaries and 200 machine (OpenAI) summaries
based on the same Wikipedia titles, then saves to dataset.csv with columns:
text, label, topic.
"""

import argparse
import csv
import os
import random
import time
import warnings
from typing import Optional

from bs4 import GuessedAtParserWarning
from dotenv import load_dotenv
import httpx
import wikipedia
from openai import OpenAI

warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY.strip() in {"", "your_openai_api_key_here"}:
    raise ValueError(
        "Please set OPENAI_API_KEY in your .env file. "
        "Get a key from https://platform.openai.com/api-keys"
    )


def _wikipedia_random_titles(pages: int) -> list[str]:
    """
    Return `pages` random Wikipedia titles.
    The wikipedia package supports either wikipedia.random(pages=n) or wikipedia.random(n) depending on version.
    """
    try:
        batch = wikipedia.random(pages=pages)
    except TypeError:
        batch = wikipedia.random(pages)

    if isinstance(batch, str):
        return [batch]
    return list(batch)


def fetch_human_summary(title: str) -> Optional[str]:
    """Fetch Wikipedia summary for a title. Returns None on failure."""
    try:
        summary = wikipedia.summary(title, auto_suggest=False, redirect=True)
        return summary.strip() if summary else None
    except (
        wikipedia.exceptions.DisambiguationError,
        wikipedia.exceptions.PageError,
        wikipedia.exceptions.WikipediaException,
    ):
        return None


def fetch_machine_summary(
    client: OpenAI,
    title: str,
    model: str,
    max_tokens: int,
    max_retries: int,
    base_sleep_s: float,
) -> Optional[str]:
    """Generate a Wikipedia-style summary using OpenAI for the given topic (with retries)."""
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write a concise, encyclopedic Wikipedia-style summary. "
                            "Return only the summary text (no headings, no preamble, no meta commentary)."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Write a brief Wikipedia-style summary for the topic: {title}",
                    },
                ],
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content if response.choices else None
            text = text.strip() if text else None
            if text:
                return text
        except Exception as e:
            last_err = e
            # Jittered exponential backoff for transient failures (rate limits, timeouts, 5xx)
            sleep_s = base_sleep_s * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, 30.0) + random.random()
            time.sleep(sleep_s)

    if last_err:
        raise last_err
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of topics/summaries per class (default: 200)")
    parser.add_argument("--output", type=str, default="dataset.csv", help="Output CSV path (default: dataset.csv)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model for machine summaries")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens for generated summary")
    parser.add_argument("--wiki-batch", type=int, default=25, help="How many random Wikipedia titles to sample per batch")
    parser.add_argument("--sleep", type=float, default=0.2, help="Base sleep between requests (seconds)")
    parser.add_argument("--openai-timeout", type=float, default=60.0, help="OpenAI request timeout (seconds)")
    parser.add_argument("--openai-retries", type=int, default=5, help="Retries per topic for OpenAI generation")
    args = parser.parse_args()

    wikipedia.set_lang("en")

    print("Loading OpenAI client...")
    timeout = httpx.Timeout(args.openai_timeout)
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=timeout)

    n = args.n
    output_path = args.output

    # Step 1: collect N valid Wikipedia summaries (human) and topics
    print(f"Collecting {n} human Wikipedia summaries...")
    human: list[dict] = []
    seen_topics: set[str] = set()
    attempts = 0
    max_attempts = n * 50
    while len(human) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"Could not collect {n} valid Wikipedia summaries after {max_attempts} attempts.")

        batch = _wikipedia_random_titles(args.wiki_batch)
        for topic in batch:
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            human_text = fetch_human_summary(topic)
            if human_text:
                human.append({"text": human_text, "label": 0, "topic": topic})
                if len(human) % 20 == 0:
                    print(f"Human progress: {len(human)}/{n}")
            time.sleep(args.sleep)
            if len(human) >= n:
                break

    topics = [r["topic"] for r in human]

    # Step 2: generate machine summaries for the same topics
    print(f"Generating {n} machine summaries for the same topics...")
    machine: list[dict] = []
    for idx, topic in enumerate(topics, start=1):
        machine_text = fetch_machine_summary(
            client=client,
            title=topic,
            model=args.model,
            max_tokens=args.max_tokens,
            max_retries=args.openai_retries,
            base_sleep_s=max(args.sleep, 0.2),
        )
        if not machine_text:
            raise RuntimeError(f"OpenAI returned empty content for topic: {topic!r}")
        machine.append({"text": machine_text, "label": 1, "topic": topic})
        if idx % 20 == 0:
            print(f"Machine progress: {idx}/{n}")
        time.sleep(args.sleep)

    final_rows = human + machine

    print(f"Writing {len(final_rows)} rows to {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "topic"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Done. Saved to {output_path}.")


if __name__ == "__main__":
    main()
