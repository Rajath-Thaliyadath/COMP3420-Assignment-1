"""
Data collection script for COMP8420 NLP assignment.
Fetches 200 human (Wikipedia) summaries and 200 machine (OpenAI) summaries
based on the same Wikipedia titles, then saves to dataset.csv with columns:
text, label, topic.
"""

import argparse
import csv
import hashlib
import html
import os
import random
import re
import time
import warnings
from typing import Optional

from bs4 import BeautifulSoup, GuessedAtParserWarning
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


_CITATION_RE = re.compile(r"\[\d+\]")
_WS_RE = re.compile(r"\s+")
_EN_LETTER_RE = re.compile(r"[A-Za-z]")


def clean_text(text: str) -> str:
    """
    Normalize text for CSV:
    - strip HTML (e.g., anchor tags) if present
    - unescape HTML entities
    - remove numeric citations like [1]
    - collapse all whitespace/newlines to single spaces
    """
    stripped = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    stripped = html.unescape(stripped)
    stripped = _CITATION_RE.sub("", stripped)
    stripped = _WS_RE.sub(" ", stripped).strip()
    return stripped


def is_mostly_english(text: str, min_ratio: float = 0.8) -> bool:
    """
    Heuristic filter to keep only mostly-English text.
    Checks:
      - high proportion of ASCII/Latin-like characters
      - at least some letters and vowels
    """
    if not text:
        return False
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:'\"!?-()")
    total = len(text)
    if total == 0:
        return False
    ascii_like = sum(1 for ch in text if ch in allowed_chars)
    if ascii_like / total < min_ratio:
        return False
    if not _EN_LETTER_RE.search(text):
        return False
    if not re.search(r"[AEIOUaeiou]", text):
        return False
    return True


def text_fingerprint(text: str) -> str:
    normalized = clean_text(text).casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def load_existing_dataset(output_path: str) -> tuple[set[str], set[str]]:
    existing_topics: set[str] = set()
    existing_text_fps: set[str] = set()
    if not os.path.exists(output_path):
        return existing_topics, existing_text_fps

    with open(output_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = (row.get("topic") or "").strip()
            text = row.get("text") or ""
            if topic:
                existing_topics.add(topic)
            if text.strip():
                existing_text_fps.add(text_fingerprint(text))

    return existing_topics, existing_text_fps


def open_dataset_writer(output_path: str, append: bool) -> tuple[csv.DictWriter, object]:
    file_exists = os.path.exists(output_path)
    mode = "a" if append and file_exists else "w"
    f = open(output_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=["text", "label", "topic"])
    if mode == "w":
        writer.writeheader()
        f.flush()
    return writer, f


def fetch_human_summary(title: str) -> Optional[str]:
    """Fetch Wikipedia summary for a title. Returns None on failure."""
    try:
        summary = wikipedia.summary(title, auto_suggest=False, redirect=True)
        if not summary:
            return None
        text = clean_text(summary)
        if not is_mostly_english(text):
            return None
        return text
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
            if not text:
                continue
            text = clean_text(text)
            if not is_mostly_english(text):
                continue
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
    parser.add_argument("--n", type=int, default=20, help="How many NEW topic pairs to add this run (default: 20)")
    parser.add_argument("--output", type=str, default="dataset.csv", help="Output CSV path (default: dataset.csv)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model for machine summaries")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens for generated summary")
    parser.add_argument("--wiki-batch", type=int, default=25, help="How many random Wikipedia titles to sample per batch")
    parser.add_argument("--sleep", type=float, default=0.2, help="Base sleep between requests (seconds)")
    parser.add_argument("--openai-timeout", type=float, default=60.0, help="OpenAI request timeout (seconds)")
    parser.add_argument("--openai-retries", type=int, default=5, help="Retries per topic for OpenAI generation")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing CSV instead of overwriting (recommended for incremental runs)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV even if it already exists",
    )
    args = parser.parse_args()

    wikipedia.set_lang("en")

    print("Loading OpenAI client...")
    timeout = httpx.Timeout(args.openai_timeout)
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=timeout)

    n = args.n
    output_path = args.output

    if args.overwrite and os.path.exists(output_path):
        os.remove(output_path)

    append = args.append and not args.overwrite
    existing_topics, existing_text_fps = load_existing_dataset(output_path)
    if existing_topics:
        print(f"Found existing dataset with {len(existing_topics)} topics. Will add {n} new topics.")
    else:
        print(f"No existing dataset found. Will create a new one with {n} topics.")

    writer, f = open_dataset_writer(output_path, append=append)

    added = 0
    attempts = 0
    max_attempts = max(500, n * 200)
    newly_added_topics: set[str] = set()
    try:
        while added < n:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Could not add {n} new topics after {max_attempts} attempts.")

            for topic in _wikipedia_random_titles(args.wiki_batch):
                topic = (topic or "").strip()
                if not topic:
                    continue

                if topic in existing_topics or topic in newly_added_topics:
                    continue

                human_text = fetch_human_summary(topic)
                if not human_text:
                    time.sleep(args.sleep)
                    continue

                human_fp = text_fingerprint(human_text)
                if human_fp in existing_text_fps:
                    time.sleep(args.sleep)
                    continue

                machine_text = fetch_machine_summary(
                    client=client,
                    title=topic,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    max_retries=args.openai_retries,
                    base_sleep_s=max(args.sleep, 0.2),
                )
                if not machine_text:
                    time.sleep(args.sleep)
                    continue

                machine_fp = text_fingerprint(machine_text)
                if machine_fp in existing_text_fps:
                    time.sleep(args.sleep)
                    continue

                writer.writerow({"text": human_text, "label": 0, "topic": topic})
                writer.writerow({"text": machine_text, "label": 1, "topic": topic})
                f.flush()

                existing_topics.add(topic)
                newly_added_topics.add(topic)
                existing_text_fps.add(human_fp)
                existing_text_fps.add(machine_fp)
                added += 1

                if added % 20 == 0 or added == n:
                    print(f"Added {added}/{n} new topics...")

                time.sleep(args.sleep)
                if added >= n:
                    break
    finally:
        f.close()

    print(f"Done. Added {added} new topics ({added} human + {added} machine) to {output_path}.")


if __name__ == "__main__":
    main()
