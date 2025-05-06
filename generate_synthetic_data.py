#!/usr/bin/env python3
import os
import random
import string
import nltk

# 1) Download Brown corpus
nltk.download('brown', quiet=True)
from nltk.corpus import brown


def load_sentences():
    """
    Load and clean Brown sentences, removing any token containing backticks.
    """
    sents = []
    for sent in brown.sents():
        # drop any token with a backtick in it
        clean_tokens = [tok for tok in sent if '`' not in tok]
        if not clean_tokens:
            continue
        # join & lowercase
        s = " ".join(clean_tokens).lower()
        sents.append(s)
    return sents

def generate_dataset(
    n=20000,
    min_len=5,
    max_len=100,
    input_path="data/input.txt",
    answer_path="data/answer.txt",
    joined_path="data/joined.txt"
):
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    sentences = load_sentences()

    inputs, targets, full_words = [], [], []
    while len(inputs) < n:
        sent = random.choice(sentences).strip()

        # ensure length >= min_len+1
        if len(sent) < min_len + 1:
            continue

        # truncate to max_len
        if len(sent) > max_len:
            sent = sent[:max_len]

        # find valid cut points (no space before, target is lowercase)
        candidates = [
            i for i in range(min_len, len(sent))
            if sent[i] in string.ascii_lowercase and sent[i-1] != ' '
        ]
        if not candidates:
            continue

        cut = random.choice(candidates)
        prefix = sent[:cut]
        ch     = sent[cut]

        # extract the full word containing ch
        start = sent.rfind(' ', 0, cut) + 1
        end   = sent.find(' ', cut)
        if end == -1:
            end = len(sent)
        word = sent[start:end]

        inputs.append(prefix)
        targets.append(ch)
        full_words.append(word)

    # write out
    with open(input_path,  "w") as f: f.write("\n".join(inputs)     + "\n")
    with open(answer_path, "w") as f: f.write("\n".join(targets)    + "\n")
    with open(joined_path, "w") as f: f.write("\n".join(full_words) + "\n")

    print(f"Wrote {len(inputs)} examples to:")
    print(f" • {input_path}")
    print(f" • {answer_path}")
    print(f" • {joined_path}")

if __name__ == "__main__":
    generate_dataset(
        n=20000,
        min_len=5,
        max_len=40,
        input_path="data/input.txt",
        answer_path="data/answer.txt",
        joined_path="data/joined.txt"
    )
