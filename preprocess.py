"""
Text preprocessing script.
Applies lowercase, punctuation removal, number removal, stopword removal,
and lemmatization to the text column, then saves with a new 'cleaned_text' column.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet

# Download required NLTK data (run once)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def get_wordnet_pos(treebank_tag: str) -> str:
    """Map NLTK treebank POS tag to WordNet POS for lemmatizer."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def preprocess_text(text: str) -> str:
    """
    Preprocess text: lowercase, remove punctuation and numbers,
    remove English stop words, and apply lemmatization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove numbers (digits)
    text = re.sub(r"\d+", "", text)

    # Remove punctuation (keep letters and spaces)
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Load English stop words and lemmatizer
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Remove stop words and lemmatize (with POS for better lemmatization)
    pos_tags = pos_tag(tokens)
    cleaned_tokens = []
    for word, tag in pos_tags:
        if word in stop_words:
            continue
        pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=pos)
        if lemma:
            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)


def main():
    # Read dataset (engine='python' for multiline quoted fields; skip bad lines)
    try:
        df = pd.read_csv(
            "dataset.csv",
            engine="python",
            on_bad_lines="skip",
        )
    except TypeError:
        # Older pandas: use error_bad_lines=False
        df = pd.read_csv("dataset.csv", engine="python", error_bad_lines=False)

    # Apply preprocessing to text column
    df["cleaned_text"] = df["text"].apply(preprocess_text)

    # Save to new file
    df.to_csv("dataset_clean_1.csv", index=False)
    print("Saved dataset_clean_1.csv with column 'cleaned_text'.")


if __name__ == "__main__":
    main()
