#!/usr/bin/env python3
import os
import random
import string

def load_common_words(path="data/top_10k_words.txt"):
    with open(path) as f:
        words = [w.strip().lower() for w in f if w.strip().isalpha()]
    if not words:
        raise ValueError(f"No valid words in {path}")
    return words


def generate_dataset(
    n=20000,
    min_len=5,
    max_len=40,
    wordlist_path="data/top_10k_words.txt",
    input_path="data/input.txt",
    answer_path="data/answer.txt",
    joined_path="data/joined.txt"
):
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    common_words = load_common_words(wordlist_path)

    inputs, targets, full_words = [], [], []

    while len(inputs) < n:
        # 1) pick a random target length for the full sentence
        target_len = random.randint(min_len + 1, max_len)

        # 2) assemble until we reach at least target_len, then clip
        sentence = ""
        while len(sentence) < target_len:
            w = random.choice(common_words)
            sentence += (" " if sentence else "") + w
            if len(sentence) >= max_len:
                sentence = sentence[:max_len]
                break

        # 3) skip if we still don't have room for a min-length prefix + 1 char
        if len(sentence) < min_len + 1:
            continue

        # 4) find valid cut-points at >= min_len where char is lowercase and prefix doesn't end with space
        candidates = [
            i for i in range(min_len, len(sentence))
            if sentence[i] in string.ascii_lowercase and sentence[i-1] != ' '
        ]
        if not candidates:
            continue

        # 5) choose a random cut
        cut = random.choice(candidates)
        prefix = sentence[:cut]
        ch     = sentence[cut]

        # 6) extract the full word containing that char
        start = sentence.rfind(' ', 0, cut) + 1
        end   = sentence.find(' ', cut)
        if end == -1:
            end = len(sentence)
        word = sentence[start:end]

        # 7) record example
        inputs.append(prefix)
        targets.append(ch)
        full_words.append(word)

    # 8) write out files
    with open(input_path,  "w") as f:
        f.write("\n".join(inputs)    + "\n")
    with open(answer_path, "w") as f:
        f.write("\n".join(targets)   + "\n")
    with open(joined_path, "w") as f:
        f.write("\n".join(full_words) + "\n")

    print(f"Wrote {len(inputs)} examples to:")
    print(f" • {input_path}")
    print(f" • {answer_path}")
    print(f" • {joined_path}")


if __name__ == "__main__":
    generate_dataset(
        n=20000,
        min_len=5,
        max_len=40,
        wordlist_path="data/top_10k_words.txt",
        input_path="data/input.txt",
        answer_path="data/answer.txt",
        joined_path="data/joined.txt"
    )
