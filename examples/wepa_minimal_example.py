"""Minimal WEPA example with tiny in-memory embeddings.

Run from the repository root:

    python examples/wepa_minimal_example.py

This example is intentionally small. It demonstrates the Word Embedding
Projection Approach (WEPA) scoring workflow without external datasets, trained
embedding files, or internet access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cntext as ct


class TinyEmbedding:
    """Small KeyedVectors-like object for examples and tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = {word: np.array(values, dtype=float) for word, values in vectors.items()}
        self.vector_size = len(next(iter(self._vectors.values())))
        self.index_to_key = list(self._vectors)
        self.key_to_index = {word: idx for idx, word in enumerate(self.index_to_key)}
        self.vectors = np.vstack([self._vectors[word] for word in self.index_to_key])

    def __contains__(self, word: str) -> bool:
        return word in self._vectors

    def __getitem__(self, word: str) -> np.ndarray:
        return self._vectors[word]

    def get_vector(self, word: str) -> np.ndarray:
        return self._vectors[word]

    def get_mean_vector(self, words: list[str]) -> np.ndarray:
        valid_vectors = [self._vectors[word] for word in words if word in self._vectors]
        if not valid_vectors:
            return np.zeros(self.vector_size)
        return np.mean(valid_vectors, axis=0)


def main() -> None:
    # A tiny toy embedding space. Positive anchors point to the right on the
    # first dimension; negative anchors point to the left.
    wv = TinyEmbedding(
        {
            "commit": [1.0, 0.0],
            "persist": [0.9, 0.1],
            "focus": [0.8, 0.0],
            "quit": [-1.0, 0.0],
            "avoid": [-0.9, 0.0],
            "delay": [-0.8, -0.1],
            "goal": [0.6, 0.2],
            "today": [0.1, 0.0],
            "maybe": [-0.1, 0.1],
        }
    )

    positive_pole = ["commit", "persist", "focus"]
    negative_pole = ["quit", "avoid", "delay"]

    axis = ct.generate_concept_axis(
        wv=wv,
        poswords=positive_pole,
        negwords=negative_pole,
    )

    texts = [
        "commit persist focus goal today",
        "maybe delay avoid goal",
        "unseen words only",
    ]

    rows = []
    for text in texts:
        score = ct.project_text(wv=wv, text=text, axis=axis, lang="english")
        rows.append({"text": text, "wepa_score": score})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

