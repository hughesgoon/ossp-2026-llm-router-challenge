from __future__ import annotations

import collections
import json
import pickle
import re

from train.paths import ARTIFACTS, DATA, ensure_artifacts

MERGES = 4000
WORD = re.compile(r"[A-Za-z']+|[가-힣]+|\d+|\S")


def main() -> int:
    episodes = json.loads(
        (DATA / "train" / "inputs-base.json").read_text("utf-8")
    )["episodes"]
    freq = collections.Counter()
    for episode in episodes:
        freq.update(WORD.findall(episode.get("prompt", "").lower()))

    vocab = {word: tuple(word) + ("</w>",) for word in freq}
    rules = []
    for _ in range(MERGES):
        pairs = collections.Counter()
        for word, symbols in vocab.items():
            count = freq[word]
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += count
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        rules.append(best)
        merged = {}
        for word, symbols in vocab.items():
            out = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    out.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            merged[word] = tuple(out)
        vocab = merged

    ensure_artifacts()
    pickle.dump({"rules": rules, "vocab": vocab}, open(ARTIFACTS / "bpe.pkl", "wb"))
    (ARTIFACTS / "bpe_rules.json").write_text(
        json.dumps([[a, b] for a, b in rules]), encoding="utf-8"
    )
    print(f"rules={len(rules)} vocab={len(vocab)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
