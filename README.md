# cntext: Text Analysis for Computational Social Science

**cntext** is a Python package for computational social science researchers who work with text data. It supports file reading, dictionary-based analysis, word frequency statistics, readability, similarity, word embeddings, semantic projection, plotting, and LLM-assisted structured text analysis.

This repository now foregrounds reproducible support for the **Word Embedding Projection Approach (WEPA)** introduced in the manuscript:

> [Measuring Psychological Constructs from Social Media Text Using the Word Embedding Projection Approach](WEPA-DEMO/01-WEPA-Intro.md)

**Reviewer entry point:** open [`WEPA-DEMO/01-WEPA-Intro.md`](WEPA-DEMO/01-WEPA-Intro.md) for the paper-oriented WEPA demo, then follow the three reproducible workflow files in [`WEPA-DEMO/`](WEPA-DEMO/).

## Word Embedding Projection Approach (WEPA)

WEPA is a **theory-driven semantic projection workflow** for measuring construct-related linguistic salience in user-generated text. It represents a psychological or social construct as a **semantic axis** in a word-embedding space. The axis is defined by theory-based **positive pole** and **negative pole** anchor words. A text is then projected onto that axis to obtain a text-based indicator.

In cntext, the shortest WEPA scoring path is one line:

```python
score = ct.wepa(
    wv=wv,
    text="I will persist and focus on this goal",
    poswords=["commit", "persist", "focus"],
    negwords=["quit", "avoid", "delay"],
    lang="english",
)
```

This one-line call is equivalent to the expanded workflow:

```python
axis = ct.generate_concept_axis(
    wv=wv,
    poswords=["commit", "persist", "focus"],
    negwords=["quit", "avoid", "delay"],
)

score = ct.project_text(
    wv=wv,
    text="I will persist and focus on this goal",
    axis=axis,
    lang="english",
)
```

WEPA scores should be interpreted as indicators of **construct-related linguistic salience** in text. They should not be interpreted as direct observations of latent psychological states, clinical diagnoses, causal effects, or proof of strict measurement invariance. Anchor-word dictionaries are context-specific measurement resources. They should be developed from theory, reviewed by domain experts, and validated empirically before being used in new platforms, languages, or cultural contexts.

## Installation

Install the latest released package:

```bash
pip install cntext --upgrade
```

Recommended Python versions: Python 3.9 to 3.12.

For local development from this repository:

```bash
pip install -e .
```

## Minimal Runnable WEPA Example

The repository includes a small example that does not require external datasets, large embedding files, or internet access:

```bash
python examples/wepa_minimal_example.py
```

If your system uses `python3`:

```bash
python3 examples/wepa_minimal_example.py
```

The example creates a tiny in-memory embedding object, defines positive and negative anchor words for a goal-commitment construct, constructs a semantic axis, scores several toy texts, and prints a small table.

## WEPA Documentation and Reproducibility Files

- Paper demo entry point: [`WEPA-DEMO/01-WEPA-Intro.md`](WEPA-DEMO/01-WEPA-Intro.md)
- Build corpus demo: [`WEPA-DEMO/2. Build-Corpus.md`](WEPA-DEMO/2.%20Build-Corpus.md)
- Train embeddings demo: [`WEPA-DEMO/3. Train-Embeddings.md`](WEPA-DEMO/3.%20Train-Embeddings.md)
- Semantic-axis and scoring demo: [`WEPA-DEMO/4. Semantic-Axis-And-Scoring.md`](WEPA-DEMO/4.%20Semantic-Axis-And-Scoring.md)
- WEPA guide: [`docs/wepa.md`](docs/wepa.md)
- Paper reproducibility guide: [`docs/paper_reproducibility.md`](docs/paper_reproducibility.md)
- Minimal example script: [`examples/wepa_minimal_example.py`](examples/wepa_minimal_example.py)
- Anchor dictionary format: [`examples/wepa_anchor_dictionary_format.md`](examples/wepa_anchor_dictionary_format.md)
- Lightweight WEPA tests: [`tests/test_wepa.py`](tests/test_wepa.py)

Run the WEPA tests:

```bash
python -m pytest tests/test_wepa.py
```

or:

```bash
python3 -m pytest tests/test_wepa.py
```

The public repository supports reproducibility of the WEPA scoring workflow. The raw platform data used in the manuscript are not redistributed because they contain user-generated content and must be protected for privacy reasons. Full reproduction of the empirical tables requires access to the original de-identified research dataset and the corresponding research-use permissions.

## Quick Start

```python
import cntext as ct

print("cntext version:", ct.__version__)
ct.hello()
```

Common workflow:

```python
import cntext as ct

text = "The product is useful, reliable, and easy to use."

dictionary_data = ct.read_yaml_dict("en_common_NRC.yaml")
sentiment_dictionary = dictionary_data["Dictionary"]

result = ct.sentiment(
    text=text,
    diction=sentiment_dictionary,
    lang="english",
)
print(result)
```

## Module Overview

cntext contains six main areas.

| Area | Main functions | Description |
| --- | --- | --- |
| `io` | `get_cntext_path`, `get_dict_list`, `read_yaml_dict`, `read_pdf`, `read_docx`, `read_file`, `read_files`, `get_files`, `extract_mda`, `traditional2simple`, `fix_text`, `fix_contractions`, `clean_text` | Read files, load dictionaries, clean text, and prepare research datasets. |
| `stats` | `word_count`, `readability`, `sentiment`, `sentiment_by_valence`, `word_in_context`, `epu`, `fepu`, `semantic_brand_score`, `cosine_sim`, `jaccard_sim`, `minedit_sim`, `word_hhi` | Compute word counts, readability, dictionary scores, similarity measures, and domain indicators. |
| `plot` | `matplotlib_chinese`, `lexical_dispersion_plot1`, `lexical_dispersion_plot2` | Create lexical dispersion plots and configure Matplotlib for CJK text rendering. |
| `model` | `Word2Vec`, `GloVe`, `FastText`, `SoPmi`, `load_w2v`, `glove2word2vec`, `expand_dictionary`, `evaluate_similarity`, `evaluate_analogy` | Train, load, evaluate, and use embedding models and dictionary expansion tools. |
| `mind` | `generate_concept_axis`, `project_text`, `wepa`, `sematic_projection`, `project_word`, `semantic_centroid`, `sematic_distance`, `procrustes_align`, `divergent_association_task`, `discursive_diversity_score` | Measure semantic projection, construct-related linguistic salience, semantic distance, embedding alignment, and related indicators. |
| `llm` | `llm`, `analysis_by_llm`, `text_analysis_by_llm` | Use LLMs for structured text analysis tasks. |

## 1. IO Module

The IO module helps researchers load files, inspect built-in dictionaries, clean text, and prepare corpora for analysis.

### 1.1 `get_dict_list()`

List the built-in YAML dictionaries.

```python
import cntext as ct

ct.get_dict_list()
```

Typical built-in dictionary files include:

- `en_common_NRC.yaml`
- `en_common_LoughranMcDonald.yaml`
- `en_common_LSD2015.yaml`
- `en_common_SentiWS.yaml`
- `en_valence_Concreteness.yaml`
- `zh_common_DUTIR.yaml`
- `zh_common_HowNet.yaml`
- `zh_common_NTUSD.yaml`
- `zh_common_EPU.yaml`
- `zh_common_FEPU.yaml`
- `zh_common_LoughranMcDonald.yaml`
- `zh_valence_ChineseEmoBank.yaml`
- `zh_valence_SixSemanticDimensionDatabase.yaml`
- `enzh_common_StopWords.yaml`
- `enzh_common_AdvConj.yaml`

### 1.2 Built-in YAML Dictionaries

cntext dictionaries are stored as YAML files. A dictionary usually contains metadata and a `Dictionary` field. For example:

```yaml
Name: Example Dictionary
Desc: Short description of the dictionary.
Refer: Reference information.
Category:
  - positive
  - negative
Dictionary:
  positive:
    - good
    - reliable
  negative:
    - bad
    - risky
```

Built-in dictionaries cover general sentiment, finance, uncertainty, stop words, rhetorical categories, concreteness, and other lexical resources. They are useful starting points, but researchers should check whether a dictionary is appropriate for their research domain.

### 1.3 `read_yaml_dict()`

Load a built-in or custom YAML dictionary.

```python
import cntext as ct

data = ct.read_yaml_dict("en_common_NRC.yaml")
dictionary = data["Dictionary"]
```

Function signature:

```python
ct.read_yaml_dict(yfile, is_builtin=True)
```

Parameters:

- `yfile`: YAML dictionary filename or path.
- `is_builtin`: if `True`, read from cntext built-in dictionaries; if `False`, read from a user-provided file path.

Returns:

- A Python dictionary containing dictionary metadata and lexical categories.

### 1.4 `detect_encoding()`

Detect the encoding of a text file.

```python
encoding = ct.detect_encoding("data/example.txt")
print(encoding)
```

Function signature:

```python
ct.detect_encoding(file, num_lines=100)
```

### 1.5 `get_files(fformat)`

Return files matching a path pattern.

```python
files = ct.get_files("data/*.txt")
```

This is useful for building file-level datasets from a folder of text, PDF, Word, or CSV files.

### 1.6 `read_pdf()`

Read text from a PDF file.

```python
text = ct.read_pdf("data/report.pdf")
```

Function signature:

```python
ct.read_pdf(file)
```

### 1.7 `read_docx()`

Read text from a Word document.

```python
text = ct.read_docx("data/report.docx")
```

Function signature:

```python
ct.read_docx(file)
```

### 1.8 `read_file()`

Read a text file.

```python
text = ct.read_file("data/example.txt", encoding="utf-8")
```

Function signature:

```python
ct.read_file(file, encoding="utf-8")
```

### 1.9 `read_files()`

Read multiple files matching a pattern and return a DataFrame.

```python
df = ct.read_files("data/*.txt", encoding="utf-8")
```

The returned DataFrame can be used as an input table for later text analysis.

### 1.10 `extract_mda()`

Extract Management Discussion and Analysis text from annual-report content when the report structure supports extraction.

```python
mda_text = ct.extract_mda(report_text)
```

Function signature:

```python
ct.extract_mda(text, kws_pattern="")
```

Notes:

- This function is designed for financial-report workflows.
- Extraction quality depends on report formatting and section headings.
- Researchers should inspect extraction results before using them in empirical analysis.

### 1.11 `traditional2simple()`

Convert Traditional Chinese text to Simplified Chinese text, or the reverse when a different mode is supplied.

```python
converted = ct.traditional2simple(text, mode="t2s")
```

Function signature:

```python
ct.traditional2simple(text, mode="t2s")
```

### 1.12 `fix_text()`

Repair garbled or inconsistent text encoding with `ftfy`.

```python
cleaned = ct.fix_text(raw_text)
```

### 1.13 `fix_contractions()`

Expand English contractions.

```python
ct.fix_contractions("you're right")
```

Output:

```text
you are right
```

### 1.14 `clean_text()`

Clean text for Chinese or English preprocessing.

```python
cleaned = ct.clean_text(text, lang="english")
```

Function signature:

```python
ct.clean_text(text, lang="chinese")
```

Supported values:

- `lang="chinese"`
- `lang="english"`

## 2. Stats Module

The Stats module provides traditional text statistics, dictionary scoring, contextual word search, uncertainty indicators, brand salience, similarity measures, and lexical concentration.

### 2.1 `word_count()`

Count words or tokens in a text.

```python
result = ct.word_count("This is a short example text.", lang="english")
```

Function signature:

```python
ct.word_count(text, lang="chinese")
```

### 2.2 `readability()`

Compute readability indicators.

```python
result = ct.readability(text, lang="english")
```

Function signature:

```python
ct.readability(text, lang="chinese", syllables=3)
```

Notes:

- English readability uses sentence and word information.
- Chinese readability depends on the package's tokenization and length assumptions.
- Readability indicators should be interpreted as descriptive features, not as direct quality scores.

### 2.3 `sentiment(text, diction, lang)`

Compute dictionary-based sentiment or category counts with equal word weights.

```python
dictionary = {
    "positive": ["good", "reliable", "useful"],
    "negative": ["bad", "risky", "weak"],
}

result = ct.sentiment(
    text="The product is reliable and useful.",
    diction=dictionary,
    lang="english",
)
```

Function signature:

```python
ct.sentiment(text, diction, lang="chinese", return_series=False)
```

Returns:

- Category counts such as `positive_num` and `negative_num`.
- Text-level counts such as `word_num`, `sentence_num`, and `stopword_num`.

### 2.4 `sentiment_by_valence()`

Compute dictionary-based scores when dictionary entries carry numeric values.

```python
valence_dictionary = {
    "word": ["good", "bad"],
    "valence": [1.0, -1.0],
}

result = ct.sentiment_by_valence(
    text="good good bad",
    diction=valence_dictionary,
    lang="english",
)
```

Function signature:

```python
ct.sentiment_by_valence(text, diction, lang="chinese", mean=False, return_series=False)
```

### 2.5 `word_in_context()`

Find keywords and return their surrounding context.

```python
contexts = ct.word_in_context(
    text="The team will commit to the goal and persist.",
    keywords=["commit", "persist"],
    window=3,
    lang="english",
)
```

Function signature:

```python
ct.word_in_context(text, keywords, window=3, lang="chinese")
```

### 2.6 `epu()`

Compute or load an Economic Policy Uncertainty indicator using the package workflow.

```python
df = ct.epu()
```

Researchers should inspect the underlying corpus, dictionary choices, and time aggregation before using EPU results in empirical models.

### 2.7 `fepu()`

Compute firm-level economic policy uncertainty perception from text.

```python
result = ct.fepu(text)
```

Function signature:

```python
ct.fepu(text, ep_pattern="", u_pattern="")
```

### 2.8 `semantic_brand_score()`

Compute Semantic Brand Score indicators for brands, organizations, individuals, or keywords.

```python
result = ct.semantic_brand_score(
    text=text,
    brands=["brand_a", "brand_b"],
    lang="english",
)
```

Function signature:

```python
ct.semantic_brand_score(text, brands, lang="chinese", co_range=7, link_filter=2)
```

The Semantic Brand Score combines prevalence, diversity, and connectivity. Researchers should choose the co-occurrence range and filtering parameters according to the corpus and research question.

### 2.9 Text Similarity

cntext includes several text similarity measures.

```python
text1 = "The company invests in innovation."
text2 = "The firm supports innovative research."

ct.cosine_sim(text1, text2, lang="english")
ct.jaccard_sim(text1, text2, lang="english")
ct.minedit_sim(text1, text2, lang="english")
ct.simple_sim(text1, text2, lang="english")
```

Functions:

```python
ct.cosine_sim(text1, text2, lang="chinese")
ct.jaccard_sim(text1, text2, lang="chinese")
ct.minedit_sim(text1, text2, lang="chinese")
ct.simple_sim(text1, text2, lang="chinese")
```

### 2.10 `word_hhi()`

Compute the Herfindahl-Hirschman Index of word concentration in a text.

```python
hhi = ct.word_hhi(text)
```

This can be used as a descriptive indicator of lexical concentration or repetition.

## 3. Plot Module

The Plot module provides helper functions for lexical dispersion visualization and CJK font support.

### 3.1 `matplotlib_chinese()`

Configure Matplotlib to display CJK text.

```python
ct.matplotlib_chinese()
```

### 3.2 `lexical_dispersion_plot1()`

Plot where target words appear within a single text.

```python
targets = {
    "positive": ["good", "strong", "reliable"],
    "negative": ["bad", "weak", "risky"],
}

ct.lexical_dispersion_plot1(
    text=text,
    targets_dict=targets,
    lang="english",
    title="Lexical dispersion",
)
```

Function signature:

```python
ct.lexical_dispersion_plot1(text, targets_dict, lang, title, figsize)
```

### 3.3 `lexical_dispersion_plot2()`

Plot target-word positions across multiple texts.

```python
texts = {
    "document_a": "The team will commit and persist.",
    "document_b": "The team may delay or avoid the goal.",
}

ct.lexical_dispersion_plot2(
    texts_dict=texts,
    targets=["commit", "persist", "delay", "avoid"],
    lang="english",
    title="Cross-document lexical dispersion",
)
```

Function signature:

```python
ct.lexical_dispersion_plot2(texts_dict, targets, lang, title, figsize)
```

## 4. Model Module

The Model module supports corpus preprocessing, embedding training, model loading, model evaluation, and dictionary expansion.

### 4.1 `Word2Vec()`

Train a Word2Vec model from a local corpus file.

```python
wv = ct.Word2Vec(
    corpus_file="examples/data/w2v_corpus.txt",
    lang="english",
    vector_size=100,
    window_size=6,
    min_count=5,
    max_iter=5,
)
```

Function signature:

```python
ct.Word2Vec(
    corpus_file,
    lang="chinese",
    dict_file=None,
    stopwords_file=None,
    vector_size=100,
    window_size=6,
    min_count=5,
    max_iter=5,
    chunksize=10000,
    only_binary=True,
    **kwargs,
)
```

Returns:

- A Gensim `KeyedVectors` model.

Notes:

- The corpus file should be plain text.
- Use a domain-relevant corpus when the model will be used for WEPA or other construct measurement.
- Store model parameters and preprocessing choices for reproducibility.

### 4.2 `GloVe()`

Train a GloVe model from a local corpus file.

```python
wv = ct.GloVe(
    corpus_file="examples/data/w2v_corpus.txt",
    lang="english",
    vector_size=100,
    window_size=15,
    min_count=5,
    max_iter=15,
)
```

Function signature:

```python
ct.GloVe(
    corpus_file,
    lang="chinese",
    dict_file=None,
    stopwords_file=None,
    vector_size=100,
    window_size=15,
    min_count=5,
    max_memory=4.0,
    max_iter=15,
    x_max=10,
    chunksize=100000,
    only_binary=True,
)
```

### 4.3 `evaluate_similarity()`

Evaluate an embedding model on word similarity data.

```python
ct.evaluate_similarity(wv)
```

Function signature:

```python
ct.evaluate_similarity(wv, file=None)
```

If `file` is `None`, cntext uses a built-in evaluation file. You can also provide a custom similarity evaluation file.

### 4.4 `evaluate_analogy()`

Evaluate an embedding model on analogy data.

```python
ct.evaluate_analogy(wv)
```

Function signature:

```python
ct.evaluate_analogy(wv, file=None)
```

If `file` is `None`, cntext uses a built-in analogy evaluation file. You can also provide a custom analogy evaluation file.

### 4.5 `SoPmi()`

Expand seed dictionaries with a co-occurrence based semantic orientation method.

```python
result = ct.SoPmi(
    corpus_file="examples/data/sopmi_corpus.txt",
    seed_file="examples/data/sopmi_seed_words.txt",
    lang="english",
)
```

Function signature:

```python
ct.SoPmi(corpus_file, seed_file, lang="chinese")
```

### 4.6 `load_w2v()`

Load a Word2Vec or GloVe model file that is compatible with cntext.

```python
wv = ct.load_w2v("path/to/model.bin")
```

Function signature:

```python
ct.load_w2v(wv_path)
```

### 4.7 `glove2word2vec()`

Convert a GloVe text model file into Word2Vec text format.

```python
ct.glove2word2vec(
    glove_file="path/to/glove.txt",
    word2vec_file="path/to/word2vec.txt",
)
```

Function signature:

```python
ct.glove2word2vec(glove_file, word2vec_file)
```

### 4.8 Notes on Embedding Models

For WEPA, embedding choice is part of the measurement design. Researchers should report:

- corpus source,
- platform and time period,
- tokenization and preprocessing,
- embedding algorithm,
- vector size,
- window size,
- minimum count threshold,
- training iterations,
- vocabulary coverage for anchor words and scored texts.

### 4.9 `expand_dictionary()`

Expand a seed dictionary with embedding neighbors.

```python
seed_dictionary = {
    "quality": ["reliable", "durable"],
    "innovation": ["novel", "creative"],
}

expanded = ct.expand_dictionary(
    wv=wv,
    seeddict=seed_dictionary,
    topn=100,
)
```

Function signature:

```python
ct.expand_dictionary(wv, seeddict, topn=100)
```

Dictionary expansion can help identify candidate words, but the expanded terms should be reviewed before use as measurement dictionaries.

## 5. Mind Module

The Mind module provides semantic projection and related embedding-based measures for social-scientific text analysis.

### 5.1 `semantic_centroid(wv, words)`

Compute the normalized semantic centroid of a word list.

```python
centroid = ct.semantic_centroid(
    wv=wv,
    words=["commit", "persist", "focus"],
)
```

Function signature:

```python
ct.semantic_centroid(wv, words)
```

Returns:

- A NumPy vector representing the centroid of valid words.

### 5.2 `generate_concept_axis(wv, poswords, negwords)`

Construct a normalized semantic axis from positive and negative pole anchor words.

```python
axis = ct.generate_concept_axis(
    wv=wv,
    poswords=["commit", "persist", "focus"],
    negwords=["quit", "avoid", "delay"],
)
```

Function signature:

```python
ct.generate_concept_axis(wv, poswords, negwords)
```

Returns:

- A unit-length NumPy vector pointing from the negative pole toward the positive pole.

Raises:

- `ValueError` if either anchor pole is empty.
- `ValueError` if the semantic axis is a zero vector.

### 5.3 `wepa(wv, text, poswords, negwords, lang, cosine)`

Score a text with the Word Embedding Projection Approach.

```python
score = ct.wepa(
    wv=wv,
    text="I will persist and focus on this goal",
    poswords=["commit", "persist", "focus"],
    negwords=["quit", "avoid", "delay"],
    lang="english",
)
```

Function signature:

```python
ct.wepa(wv, text, poswords, negwords, lang="chinese", cosine=False)
```

Interpretation:

- Higher scores indicate stronger alignment with the positive pole.
- Lower scores indicate stronger alignment with the negative pole.
- Scores indicate construct-related linguistic salience in text, not direct latent-state measurement.

### 5.4 `project_text(wv, text, axis, lang, cosine)`

Project a text onto an existing semantic axis.

```python
axis = ct.generate_concept_axis(wv, ["commit"], ["quit"])
score = ct.project_text(
    wv=wv,
    text="commit to the goal",
    axis=axis,
    lang="english",
)
```

Function signature:

```python
ct.project_text(wv, text, axis, lang="chinese", cosine=False)
```

Returns:

- The average projection score for valid in-vocabulary tokens.
- `numpy.nan` when no tokens can be scored.

### 5.5 `sematic_projection()`

Compute semantic projection scores for a list of words. The function name preserves the current public API spelling.

```python
scores = ct.sematic_projection(
    wv=wv,
    words=["mouse", "horse", "elephant"],
    poswords=["large", "big", "huge"],
    negwords=["small", "little", "tiny"],
)
```

Function signature:

```python
ct.sematic_projection(wv, words, poswords, negwords, cosine=False, return_full=True)
```

### 5.6 `project_word()`

Project one word or word list onto another word, word list, or vector.

```python
score = ct.project_word(
    wv=wv,
    a="engineer",
    b=["science", "technology"],
)
```

Function signature:

```python
ct.project_word(wv, a, b, cosine=False)
```

### 5.7 `sematic_distance()`

Compute semantic distance between two word groups. The function name preserves the current public API spelling.

```python
distance = ct.sematic_distance(
    wv=wv,
    words1=["program", "software", "computer"],
    words2=["family", "home", "parent"],
)
```

Function signature:

```python
ct.sematic_distance(wv, words1, words2)
```

### 5.8 `divergent_association_task()`

Compute a Divergent Association Task style score from a list of words.

```python
score = ct.divergent_association_task(
    wv=wv,
    words=["book", "cloud", "machine", "river", "music", "stone", "garden"],
)
```

Function signature:

```python
ct.divergent_association_task(wv, words, minimum=7)
```

### 5.9 `discursive_diversity_score()`

Compute a discursive diversity score from a list of words.

```python
score = ct.discursive_diversity_score(
    wv=wv,
    words=["strategy", "market", "team", "learning"],
)
```

Function signature:

```python
ct.discursive_diversity_score(wv, words)
```

### 5.10 `procrustes_align()`

Align two embedding spaces with a Procrustes transformation.

```python
aligned_wv = ct.procrustes_align(
    base_wv=base_wv,
    other_wv=other_wv,
)
```

Function signature:

```python
ct.procrustes_align(base_wv, other_wv, words=None)
```

This can be useful for studying semantic change or temporal comparability, but alignment alone does not prove measurement stability. Longitudinal comparability requires additional validation.

## 6. LLM Module

The LLM module supports structured text analysis with large language models.

### 6.1 `ct.llm()`

Run an LLM-assisted text analysis task.

```python
result = ct.llm(
    text="The user reports a clear goal and strong commitment.",
    prompt="Extract the main topic and sentiment.",
    output_format={"topic": "string", "sentiment": "string"},
    task="structured_analysis",
    backend="openai",
    model_name="gpt-4o-mini",
)
```

Function signature:

```python
ct.llm(
    text,
    prompt=None,
    output_format=None,
    task=None,
    backend=None,
    base_url=None,
    api_key=None,
    model_name=None,
    temperature=0,
)
```

Notes:

- LLM results should be validated for the specific task.
- For research use, report prompts, models, decoding settings, and validation checks.
- LLM outputs should not be treated as ground truth without human or empirical validation.

### 6.2 Built-in Prompts

cntext includes prompt templates for common text-analysis tasks. Use them as starting points and adapt them to your research setting.

Example workflow:

```python
prompt = "Classify the sentiment of the text as positive, neutral, or negative."

result = ct.llm(
    text="The service is useful but sometimes unstable.",
    prompt=prompt,
    output_format={"sentiment": "string"},
    task="sentiment",
)
```

## Anchor Dictionary Format for WEPA

A WEPA anchor dictionary should document the construct and both semantic poles.

Recommended JSON format:

```json
{
  "construct": "goal_commitment",
  "description": "Toy anchors for demonstrating a goal commitment semantic axis.",
  "language": "english",
  "positive_pole": ["commit", "persist", "focus"],
  "negative_pole": ["quit", "avoid", "delay"],
  "notes": "Example only. Not validated for empirical use."
}
```

Recommended CSV format:

```csv
construct,pole,anchor
goal_commitment,positive,commit
goal_commitment,positive,persist
goal_commitment,positive,focus
goal_commitment,negative,quit
goal_commitment,negative,avoid
goal_commitment,negative,delay
```

See [`examples/wepa_anchor_dictionary_format.md`](examples/wepa_anchor_dictionary_format.md) for details.

## Responsible Use

For WEPA and other construct-scoring workflows:

- Report the corpus, platform, time period, preprocessing steps, embedding model, anchor dictionaries, and validation procedures.
- Treat scores as text-based indicators, not as direct measurement of latent psychological states.
- Do not use scores as clinical diagnoses or causal evidence.
- Validate anchor dictionaries before applying them to new platforms, languages, time periods, or cultural contexts.
- Evaluate measurement stability before making longitudinal comparability claims.
- Avoid claiming universal generalizability or strict measurement invariance without dedicated evidence.

## Documentation

The Sphinx documentation is in [`docs/`](docs/). Important entry points:

- Introduction: [`docs/intro.md`](docs/intro.md)
- Installation: [`docs/install.md`](docs/install.md)
- Quick start: [`docs/quickstart.md`](docs/quickstart.md)
- IO: [`docs/io.md`](docs/io.md)
- Statistics: [`docs/stats.md`](docs/stats.md)
- Models and embeddings: [`docs/model.md`](docs/model.md), [`docs/embeddings.md`](docs/embeddings.md)
- Semantic projection and WEPA: [`docs/mind.md`](docs/mind.md), [`docs/wepa.md`](docs/wepa.md)
- LLM tools: [`docs/llm.md`](docs/llm.md)
- Plotting: [`docs/plot.md`](docs/plot.md)
- Citation: [`docs/cite.md`](docs/cite.md)

## Citation

Package citation:

```bibtex
@software{cntext,
  author = {Deng, Da},
  title = {cntext: Text Analysis Tools for Computational Social Science},
  url = {https://github.com/hiDaDeng/cntext},
  year = {2025}
}
```

WEPA manuscript citation placeholder:

```bibtex
@article{deng_wepa_forthcoming,
  author = {Deng, Da},
  title = {Measuring Psychological Constructs from Social Media Text Using the Word Embedding Projection Approach},
  year = {forthcoming}
}
```

## License

cntext is released under the MIT License. See [`LICENSE`](LICENSE).
