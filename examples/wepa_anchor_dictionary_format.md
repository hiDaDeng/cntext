# WEPA Anchor Dictionary Format

Anchor-word dictionaries define the positive pole and negative pole of a construct. They should be developed from theory, reviewed by domain experts, and validated empirically before use in a new platform, language, or cultural context.

## Recommended JSON Format

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

## Recommended CSV Format

```csv
construct,pole,anchor
goal_commitment,positive,commit
goal_commitment,positive,persist
goal_commitment,positive,focus
goal_commitment,negative,quit
goal_commitment,negative,avoid
goal_commitment,negative,delay
```

## Field Meanings

- `construct`: short construct name used in analysis files.
- `positive_pole`: anchor words that represent the higher or positive direction of the semantic axis.
- `negative_pole`: anchor words that represent the lower or negative direction of the semantic axis.
- `language`: language or language variety for the anchors.
- `notes`: source, review status, translation decisions, or known limitations.

## Translation and Localization

Anchor dictionaries should not be translated word by word without review. Translation can change connotation, usage frequency, platform meaning, and cultural salience. For multilingual or cross-platform research, create localized anchor sets and document the review process.

Anchor dictionaries from one domain should not be blindly transferred to another domain. A dictionary built for one social media platform, time period, or population may not preserve measurement stability or longitudinal comparability elsewhere.

