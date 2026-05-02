# Paper Reproducibility Guide

This guide describes what the public cntext repository supports for the revised manuscript, "Measuring Psychological Constructs from Social Media Text Using the Word Embedding Projection Approach."

## What Can Be Reproduced

The public package supports reproducibility of the WEPA scoring workflow:

1. Load or train a word-embedding model.
2. Define a construct with positive pole and negative pole anchor words.
3. Score user-provided text with the one-line wrapper `ct.wepa`.
4. Optionally construct and inspect the semantic axis with `ct.generate_concept_axis` and `ct.project_text`.
5. Inspect basic behavior with deterministic toy examples and tests.

The minimal example at `examples/wepa_minimal_example.py` demonstrates these steps without requiring the original platform data or large embedding files.

## What Cannot Be Fully Reproduced From This Repository Alone

The raw Keep platform data used in the paper are not redistributed because they contain user-generated text and must be protected for privacy reasons. Full reproduction of the empirical tables requires access to the original de-identified research dataset and the corresponding research-use permissions.

The package also does not provide universal validated dictionaries for all constructs, languages, cultures, or platforms. Anchor dictionaries used in a specific study should be documented, reviewed, and validated for that setting.

## Reproducing the WEPA Scoring Workflow

From the repository root, run the minimal example:

```bash
python examples/wepa_minimal_example.py
```

For user-provided data, the workflow is:

```python
import cntext as ct

wv = ct.load_w2v("path/to/embedding-model.bin")

positive_pole = ["commit", "persist", "focus"]
negative_pole = ["quit", "avoid", "delay"]

score = ct.wepa(
    wv=wv,
    text="I will persist and focus on this goal",
    poswords=positive_pole,
    negwords=negative_pole,
    lang="english",
)
```

The equivalent expanded form is:

```python
axis = ct.generate_concept_axis(wv, positive_pole, negative_pole)
score = ct.project_text(
    wv=wv,
    text="I will persist and focus on this goal",
    axis=axis,
    lang="english",
)
```

## Anchor Dictionary Placement

Small anchor dictionaries can be stored in project-specific files, for example:

```text
examples/data/wepa_goal_commitment_anchors.json
```

or in a study-specific folder outside the package, for example:

```text
paper_reproduction/anchors/goal_commitment.json
```

See `examples/wepa_anchor_dictionary_format.md` for recommended JSON and CSV formats.

## Tests Included

The lightweight WEPA tests use tiny deterministic vectors. They check that:

- semantic-axis construction returns the expected shape,
- positive-pole text scores higher than negative-pole text,
- empty or out-of-vocabulary text is handled gracefully,
- missing anchor-word poles raise a clear error.

Run them with:

```bash
pytest tests/test_wepa.py
```

## Responsible Interpretation

WEPA scores are text-based indicators of construct-related linguistic salience. They do not directly observe latent psychological states and should not be treated as clinical diagnoses, causal effects, or universal evidence of construct validity.

Claims about measurement stability, longitudinal comparability, or transfer across platforms require additional evidence. Researchers should report the corpus, embedding model, preprocessing steps, anchor dictionaries, validation procedures, and any platform-specific limitations.
