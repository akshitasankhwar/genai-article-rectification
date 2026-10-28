# AI Article Rectification — Solution

```bash
pip install -r requirements.txt     # Python 3.10+
cp env.example .env                 # set LLM_API_KEY, LLM_API_BASE, LLM_MODEL_NAME
python rectifier.py rectify-all     # writes all 104 files to rectified_articles/
```

No third-party services are used besides the provided LLM (`openai/gpt-oss-120b`).

## Approach

**The LLM never rewrites the article.** It only proposes small `find → replace` edits as JSON, and Python applies
them. Text that isn't targeted is copied unchanged, so correct content can't drift.

1. **Clean.** Strip the leaked `Error Annotations` block (present in 101 of 104 articles) and keep it as hints.
2. **Detect.** The LLM compares the article with the source and returns minimal edits, each with a supporting quote.
3. **Validate.** Python drops any edit whose text isn't in the article, whose quote, new words or new numbers aren't
   in the source, that is only cosmetic, that deletes more than 150 characters, or that overlaps another edit.
4. **Apply.** Accepted edits are spliced in.
5. **Verify.** A second LLM call undoes unsupported edits and adds fixes the first pass missed, validated the same way.

Any failure falls back to the cleaned article, so every article always gets an output file. Four articles run in parallel.

## Key assumptions

- The source is the only ground truth: a claim is changed only if the source contradicts it.
- Errors are local (a wrong word, number, name or clause), so fixes are word-level, never sentence rewrites.
- The annotations point at real errors but are incomplete and sometimes inexact, so they are hints, not answers.
- When unsure, leave the text unchanged: modifying correct text is penalised.

## Design choices

- **Edits, not rewrites**, so preservation is guaranteed by construction.
- **Deterministic checks against the source**, so the model can't invent fixes.
- **Verify pass**, which raised recall from 50.5% to 57.7% on the examples; prompt tuning brought it to 63.1%.
- **Medium reasoning effort.** High was about 4× slower with no overall gain.
- **Budget.** About 2 calls per article; a full run uses about 0.9M tokens (≈$0.29). All development used $0.88 of $3.

## Results

| Metric | Result |
|---|---|
| Annotated errors removed, all 104 articles | 94.5% (326/345) |
| Human fixes located / reproduced exactly (10 examples) | 63.1% / 40.5% |
| Precision of our edits (10 examples) | 81.7% |

Some human fixes use facts the source doesn't contain (for example the last paragraph of article 004), which a
source-grounded system can't reproduce.

## Files

| File | Purpose |
|---|---|
| `rectifier.py` | Entry point (`test`, `rectify-all`) |
| `rectification_system.py` | The pipeline: cleaning, LLM client, edit validation, prompts |
| `eval.py` | Development scoring: `python eval.py -v` (vs the 10 examples), `python eval.py --hints` (all 104) |
| `eval/gold/` | Copy of the 10 human-rectified examples, since `rectify-all` overwrites `rectified_articles/` |

Optional `.env` settings: `LLM_REASONING_EFFORT` (default `medium`), `RECTIFIER_WORKERS` (`4`),
`LLM_CACHE` (`1` caches replies in `.cache/`; `0` disables).
