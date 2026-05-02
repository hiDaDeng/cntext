import numpy as np
import pytest

from cntext.mind import generate_concept_axis, project_text, wepa


class TinyEmbedding:
    def __init__(self):
        self._vectors = {
            "commit": np.array([1.0, 0.0]),
            "persist": np.array([0.9, 0.1]),
            "focus": np.array([0.8, 0.0]),
            "quit": np.array([-1.0, 0.0]),
            "avoid": np.array([-0.9, 0.0]),
            "delay": np.array([-0.8, -0.1]),
            "goal": np.array([0.6, 0.1]),
        }
        self.vector_size = 2
        self.index_to_key = list(self._vectors)
        self.key_to_index = {word: idx for idx, word in enumerate(self.index_to_key)}
        self.vectors = np.vstack([self._vectors[word] for word in self.index_to_key])

    def __contains__(self, word):
        return word in self._vectors

    def __getitem__(self, word):
        return self._vectors[word]

    def get_vector(self, word):
        return self._vectors[word]

    def get_mean_vector(self, words):
        valid = [self._vectors[word] for word in words if word in self._vectors]
        if not valid:
            return np.zeros(self.vector_size)
        return np.mean(valid, axis=0)


def test_generate_concept_axis_returns_expected_shape():
    wv = TinyEmbedding()

    axis = generate_concept_axis(
        wv,
        poswords=["commit", "persist", "focus"],
        negwords=["quit", "avoid", "delay"],
    )

    assert axis.shape == (2,)
    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert axis[0] > 0


def test_wepa_scores_positive_text_higher_than_negative_text():
    wv = TinyEmbedding()
    poswords = ["commit", "persist", "focus"]
    negwords = ["quit", "avoid", "delay"]

    positive_score = wepa(wv, "commit persist focus goal", poswords, negwords, lang="english")
    negative_score = wepa(wv, "quit avoid delay", poswords, negwords, lang="english")

    assert positive_score > negative_score


def test_project_text_handles_empty_or_oov_text_gracefully():
    wv = TinyEmbedding()
    axis = generate_concept_axis(wv, ["commit"], ["quit"])

    assert np.isnan(project_text(wv, "", axis, lang="english"))
    assert np.isnan(project_text(wv, "unknown missing", axis, lang="english"))


def test_generate_concept_axis_requires_both_anchor_poles():
    wv = TinyEmbedding()

    with pytest.raises(ValueError, match="poswords"):
        generate_concept_axis(wv, [], ["quit"])

    with pytest.raises(ValueError, match="negwords"):
        generate_concept_axis(wv, ["commit"], [])

