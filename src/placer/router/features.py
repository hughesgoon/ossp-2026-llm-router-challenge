from __future__ import annotations

import json
import math
import re
from pathlib import Path

FEATURE_VERSION = 2
TOKEN_BINS = 4096
CHAR_BINS = 2048
CHAR_CUT = 500
CHAR_NGRAMS = (2, 3, 4, 5)

DENSE_NAMES = (
    "log_chars",
    "log_words",
    "hangul_ratio",
    "digit_ratio",
    "has_code",
    "has_math",
    "mcq_options",
    "log_newlines",
)
FEATURE_DIM = TOKEN_BINS + CHAR_BINS + len(DENSE_NAMES)

WORD = re.compile(r"[A-Za-z']+|[가-힣]+|\d+|\S")
_CODE = re.compile(r"def \w+\(|import \w+|class \w+|```")
_MATH = re.compile(r"\\frac|\\sqrt|\\boxed|\\sum|\\int")
_MCQ = re.compile(r"(?m)^\s*([A-J])[.)]\s")

_MIX = 2654435761
_PAIR = 1000003


def _fold(value: int) -> int:
    return (value * _MIX) & 0xFFFFFFFF


class Tokenizer:
    def __init__(self, rules: list[tuple[str, str]]):
        self.rank = {(a, b): i for i, (a, b) in enumerate(rules)}
        self.cache: dict[str, tuple[str, ...]] = {}
        self.ids: dict[str, int] = {}
        for index, (a, b) in enumerate(rules):
            self.ids.setdefault(a, len(self.ids))
            self.ids.setdefault(b, len(self.ids))
            self.ids.setdefault(a + b, len(self.ids))

    def _encode_word(self, word: str) -> tuple[str, ...]:
        cached = self.cache.get(word)
        if cached is not None:
            return cached
        symbols = tuple(word) + ("</w>",)
        while True:
            best = None
            position = -1
            for i, pair in enumerate(zip(symbols, symbols[1:])):
                rank = self.rank.get(pair)
                if rank is not None and (best is None or rank < best):
                    best = rank
                    position = i
            if best is None:
                break
            symbols = (
                symbols[:position]
                + (symbols[position] + symbols[position + 1],)
                + symbols[position + 2:]
            )
        self.cache[word] = symbols
        return symbols

    def token_ids(self, prompt: str) -> list[int]:
        out = []
        for word in WORD.findall(prompt.lower()):
            for piece in self._encode_word(word):
                index = self.ids.get(piece)
                if index is None:
                    index = _fold(sum(ord(c) * (131 ** i) for i, c in enumerate(piece)))
                    index = 900000 + (index % 100000)
                    self.ids[piece] = index
                out.append(index)
        return out


def load_tokenizer(path: Path) -> Tokenizer:
    rules = [tuple(pair) for pair in json.loads(Path(path).read_text("utf-8"))]
    return Tokenizer(rules)


def _normalize(bins: list[float]) -> list[float]:
    total = 0.0
    for i, value in enumerate(bins):
        if value:
            signed = math.copysign(math.log1p(abs(value)), value)
            bins[i] = signed
            total += signed * signed
    if total > 0.0:
        scale = math.sqrt(total)
        return [value / scale for value in bins]
    return bins


def token_block(ids: list[int]) -> list[float]:
    bins = [0.0] * TOKEN_BINS
    for value in ids:
        h = _fold(value)
        bins[h & (TOKEN_BINS - 1)] += 1.0 if (h >> 20) & 1 else -1.0
    for a, b in zip(ids, ids[1:]):
        h = _fold(a * _PAIR + b)
        bins[h & (TOKEN_BINS - 1)] += 1.0 if (h >> 20) & 1 else -1.0
    return _normalize(bins)


def char_block(prompt: str) -> list[float]:
    data = prompt[:CHAR_CUT].encode("utf-8")
    bins = [0.0] * CHAR_BINS
    limit = len(data)
    for n in CHAR_NGRAMS:
        for i in range(limit - n + 1):
            h = 0
            for j in range(n):
                h = (h * 257 + data[i + j]) & 0xFFFFFFFF
            h = _fold(h)
            bins[h & (CHAR_BINS - 1)] += 1.0 if (h >> 20) & 1 else -1.0
    return _normalize(bins)


def dense_block(prompt: str) -> list[float]:
    chars = len(prompt)
    hangul = sum(1 for c in prompt if "\uac00" <= c <= "\ud7a3")
    digits = sum(1 for c in prompt if c.isdigit())
    return [
        math.log1p(chars),
        math.log1p(len(prompt.split())),
        hangul / chars if chars else 0.0,
        digits / chars if chars else 0.0,
        1.0 if _CODE.search(prompt) else 0.0,
        1.0 if _MATH.search(prompt) else 0.0,
        float(len(set(_MCQ.findall(prompt)))),
        math.log1p(prompt.count("\n")),
    ]


def extract(prompt: str, tokenizer: Tokenizer) -> list[float]:
    return (
        token_block(tokenizer.token_ids(prompt))
        + char_block(prompt)
        + dense_block(prompt)
    )
