# Word Embedding Projection Approach (WEPA)

## Overview

The Word Embedding Projection Approach (WEPA) is a theory-driven semantic projection workflow for measuring construct-related linguistic salience in text. It represents a psychological or social construct as a semantic axis in a word-embedding space, then projects user-generated text onto that axis.

WEPA is motivated by Conceptual Space Theory: a construct can be represented by direction in a semantic space when the positive pole and negative pole are defined by carefully selected anchor words. In cntext, WEPA is supported by functions for constructing semantic axes and scoring texts with those axes.

WEPA scores should be interpreted as text-based indicators. They reflect how strongly a text expresses meanings associated with a theory-defined semantic axis. They should not be interpreted as direct observations of latent psychological states, clinical diagnoses, causal effects, or proof of strict measurement invariance.

## Conceptual Workflow

1. Prepare domain text

   Collect and clean the user-generated text that matches the research setting. Preprocessing should be documented because tokenization, spelling, platform conventions, and language variety can affect the score.

2. Load or train embeddings

   Use embeddings that are appropriate for the corpus, language, time period, and platform. cntext can train Word2Vec and GloVe models, and can load compatible word-vector files.

3. Define positive and negative anchor words

   Select anchor words from theory and domain knowledge. Each construct should have a positive pole and a negative pole. Anchor dictionaries are measurement resources and should be reviewed by domain experts.

4. Construct semantic axes

   Use `ct.generate_concept_axis(wv, poswords, negwords)` to compute a normalized axis:

   ```text
   mean(positive pole anchors) - mean(negative pole anchors)
   ```

5. Project text vectors onto axes

   For most use cases, use the one-line wrapper `ct.wepa(wv, text, poswords, negwords, lang="english")`. It constructs the semantic axis and projects the text in one call. Use `ct.generate_concept_axis` and `ct.project_text` separately when you want to inspect or reuse the axis.

6. Aggregate or analyze construct scores

   Scores can be analyzed at the text, user, group, or time-window level when the research design supports that aggregation. Longitudinal comparability requires attention to measurement stability, vocabulary drift, platform changes, and consistent preprocessing.

## Core Concepts

**Anchor words**

Anchor words are theory-driven words that define the poles of a construct. For example, a goal-commitment axis might use positive pole anchors such as `commit`, `persist`, and `focus`, and negative pole anchors such as `quit`, `avoid`, and `delay`.

**Semantic axis**

A semantic axis is a direction in embedding space. In cntext, it is computed as the difference between the mean positive pole vector and the mean negative pole vector, then normalized to unit length.

**Text vector**

cntext scores a text by preprocessing it, keeping tokens found in the embedding vocabulary, and projecting each valid token onto the semantic axis.

**Projection score**

A projection score is the average projection of a text's valid token vectors onto the semantic axis. Larger values indicate that the text is more semantically aligned with the positive pole than with the negative pole, relative to the embedding model.

**Construct-related linguistic salience**

Construct-related linguistic salience means that the language in a text is more or less aligned with a construct's semantic axis. It is a text-based indicator and should be interpreted within a validated research design.

## Minimal Example

```python
import cntext as ct

poswords = ["commit", "persist", "focus"]
negwords = ["quit", "avoid", "delay"]

score = ct.wepa(
    wv=wv,
    text="I will persist and focus on today's goal",
    poswords=poswords,
    negwords=negwords,
    lang="english",
)
```

The equivalent expanded form is:

```python
axis = ct.generate_concept_axis(wv, poswords, negwords)
score = ct.project_text(
    wv=wv,
    text="I will persist and focus on today's goal",
    axis=axis,
    lang="english",
)
```

A complete runnable toy example is available at `examples/wepa_minimal_example.py`. It uses a tiny in-memory embedding object and does not require external data or internet access.

Run it from the repository root:

```bash
python examples/wepa_minimal_example.py
```

## Input and Output Format

The WEPA functions expect:

- `wv`: a Gensim `KeyedVectors` object or a compatible object with vector lookup and `get_mean_vector`.
- `poswords`: a non-empty list of positive pole anchor words.
- `negwords`: a non-empty list of negative pole anchor words.
- `text`: a string containing the text to score.
- `lang`: `"chinese"` or `"english"` for cntext preprocessing.

The output is a floating-point score. If no tokens in the text are found in the embedding vocabulary, `ct.project_text` and `ct.wepa` return `numpy.nan`.

Use `ct.wepa` when you want a compact one-line score. Use `ct.generate_concept_axis` plus `ct.project_text` when you need to store the axis, score many texts against the same axis manually, or inspect intermediate vectors.

## Limitations

WEPA depends on the quality and fit of the embedding model, the anchor-word dictionary, and the preprocessing pipeline. Scores can change when the corpus, platform, language variety, time period, or tokenizer changes.

The public cntext package supports the scoring workflow, but it does not provide universal validated dictionaries for all constructs, languages, cultures, or platforms. Anchor-word dictionaries from one domain should not be blindly transferred to another domain.

WEPA does not establish causal effects. It also does not guarantee measurement stability or longitudinal comparability by itself. Those claims require separate validation evidence.

## Responsible Interpretation

WEPA scores should be interpreted as indicators of construct-related linguistic salience in text. They reflect how strongly a text expresses meanings associated with a theory-defined semantic axis.

Anchor-word dictionaries are domain-specific measurement resources. They should be developed from theory, reviewed by domain experts, and validated empirically before being used in new platforms, languages, or cultural contexts.
