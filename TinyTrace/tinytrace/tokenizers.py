from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class NumericTokenizer:
    vocab: tuple[str, ...]
    width: int

    def __post_init__(self) -> None:
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, values: list[float]) -> list[int]:
        tokens: list[int] = []
        for index, value in enumerate(values):
            if not math.isfinite(float(value)):
                raise ValueError(f"Numeric value must be finite, received {value!r}.")
            formatted = format(value, f"0>{self.width}.1f")
            if len(formatted) != self.width or any(character not in self.token_to_id for character in formatted):
                raise ValueError(
                    f"Numeric value {value!r} cannot be represented in fixed width {self.width}."
                )
            for char in formatted:
                tokens.append(self.token_to_id[char])
            if index < len(values) - 1:
                tokens.append(self.token_to_id["<sep>"])
        tokens.append(self.token_to_id["<sync>"])
        return tokens

    def decode_sequence(self, ids: list[int]) -> list[float]:
        values: list[float] = []
        current = []
        for idx in ids:
            token = self.id_to_token[idx]
            if token == "<sync>":
                if current:
                    values.append(float("".join(current)))
                break
            if token == "<sep>":
                if current:
                    values.append(float("".join(current)))
                    current = []
            else:
                current.append(token)
        return values

    def decode_values(self, ids: list[int]) -> list[float]:
        values: list[float] = []
        current = []
        for idx in ids:
            token = self.id_to_token[idx]
            if token == "<sep>":
                if current:
                    values.append(float("".join(current)))
                    current = []
            elif token != "<sync>":
                current.append(token)
        if current:
            values.append(float("".join(current)))
        return values


class CharTokenizer:
    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [min(ord(char), self.vocab_size - 1) for char in text]

    def decode(self, ids: list[int]) -> str:
        chars = []
        for idx in ids:
            if idx <= 2:
                continue
            chars.append(chr(idx))
        return "".join(chars)
