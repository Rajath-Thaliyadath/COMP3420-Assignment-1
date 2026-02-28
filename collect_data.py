"""
Data collection script for COMP8420 NLP assignment.
Fetches 200 human (Wikipedia) summaries and 200 machine (OpenAI) summaries,
then saves to dataset.csv with columns: text, label, topic.
"""

import os
import csv
import time
from dotenv import load_dotenv
import wikipedia
from openai import OpenAI

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
    raise ValueError(
        "Please set OPENAI_API_KEY in your .env file. "
        "Get a key from https://platform.openai.com/api-keys"
    )

NUM_SUMMARIES = 200
OUTPUT_FILE = "dataset.csv"


def get_wikipedia_titles(count: int) -> list[str]:
    """Fetch a list of Wikipedia page titles (with retries for enough valid ones)."""
    titles = []
    seen = set()
    batch_size = 20
    while len(titles) < count:
        try:
            # wikipedia.random(pages=n) returns a list of n titles (or single string for n=1)
            batch = wikipedia.random(pages=min(batch_size, count - len(titles) + 5))
            if isinstance(batch, str):
                batch = [batch]
            for t in batch:
                if t and t.strip() and t not in seen:
                    seen.add(t)
                    titles.append(t)
                    if len(titles) >= count:
                        return titles
        except Exception:
            pass
    return titles[:count]


def fetch_human_summary(title: str) -> str | None:
    """Fetch Wikipedia summary for a title. Returns None on failure."""
    try:
        wikipedia.set_lang("en")
        summary = wikipedia.summary(title, auto_suggest=False, redirect=True)
        return summary.strip() if summary else None
    except (
        wikipedia.exceptions.DisambiguationError,
        wikipedia.exceptions.PageError,
        wikipedia.exceptions.WikipediaException,
    ):
        return None


def fetch_machine_summary(client: OpenAI, title: str) -> str | None:
    """Generate a Wikipedia-style summary using OpenAI for the given topic."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that writes concise, "
                        "encyclopedic Wikipedia-style summaries. Write only the summary, "
                        "no preamble or meta-commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Write a brief Wikipedia-style summary for the topic: {title}",
                },
            ],
            max_tokens=500,
        )
        text = response.choices[0].message.content
        return text.strip() if text else None
    except Exception:
        return None


def main():
    print("Loading OpenAI client...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"Fetching {NUM_SUMMARIES} Wikipedia titles...")
    titles = get_wikipedia_titles(NUM_SUMMARIES)
    print(f"Got {len(titles)} titles.")

    rows = []
    human_done = 0
    machine_done = 0

    for i, topic in enumerate(titles):
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{len(titles)} titles processed...")

        # Human (Wikipedia) summary - Label 0
        human_text = fetch_human_summary(topic)
        if human_text:
            rows.append({"text": human_text, "label": 0, "topic": topic})
            human_done += 1

        # Machine (OpenAI) summary - Label 1
        machine_text = fetch_machine_summary(client, topic)
        if machine_text:
            rows.append({"text": machine_text, "label": 1, "topic": topic})
            machine_done += 1

        # Small delay to avoid rate limits
        time.sleep(0.3)

    # If we didn't get exactly 200 of each, report and keep going (or retry).
    # Requirement: 200 human + 200 machine. We'll collect until we have 200 of each.
    needed_human = NUM_SUMMARIES - human_done
    needed_machine = NUM_SUMMARIES - machine_done

    if needed_human > 0 or needed_machine > 0:
        print(
            f"First pass: {human_done} human, {machine_done} machine. "
            f"Fetching more titles to reach 200 each..."
        )
        extra_titles = get_wikipedia_titles(needed_human + needed_machine + 50)
        seen_topics = {r["topic"] for r in rows}
        for topic in extra_titles:
            if topic in seen_topics:
                continue
            if human_done < NUM_SUMMARIES:
                human_text = fetch_human_summary(topic)
                if human_text:
                    rows.append({"text": human_text, "label": 0, "topic": topic})
                    human_done += 1
                    seen_topics.add(topic)
            if machine_done < NUM_SUMMARIES:
                machine_text = fetch_machine_summary(client, topic)
                if machine_text:
                    rows.append({"text": machine_text, "label": 1, "topic": topic})
                    machine_done += 1
                    seen_topics.add(topic)
            time.sleep(0.3)
            if human_done >= NUM_SUMMARIES and machine_done >= NUM_SUMMARIES:
                break

    # Keep only 200 human and 200 machine for a clean dataset
    human_rows = [r for r in rows if r["label"] == 0][:NUM_SUMMARIES]
    machine_rows = [r for r in rows if r["label"] == 1][:NUM_SUMMARIES]
    final_rows = human_rows + machine_rows

    print(f"Writing {len(final_rows)} rows ({len(human_rows)} human, {len(machine_rows)} machine) to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "topic"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Done. Saved to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
