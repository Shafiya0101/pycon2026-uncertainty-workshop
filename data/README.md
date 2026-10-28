# Data

Each file is JSON Lines: one question per line, 200 questions per setting.

| Folder | Content |
|---|---|
| `precomputed/` | **REAL** results exported from the thesis pipeline (put them here) |
| `synthetic/` | **FAKE** placeholders from `scripts/make_synthetic_data.py`, only so the notebooks run before the export |

If `precomputed/<setting>.jsonl` exists it is used, otherwise the synthetic file is used and every notebook prints a warning.

Settings: `mistral_triviaqa`, `qwen2_7b_triviaqa`, `qwen2_1_5b_triviaqa`, `mistral_nq`, `mistral_hotpotqa`.

## Schema

| Field | Type | Meaning |
|---|---|---|
| `qid` | str | question id |
| `question` | str | question text |
| `gold` | list[str] | accepted answers (aliases) |
| `greedy` | str | greedy answer (T=0) |
| `samples` | list[str] (10) | sampled answers (T=0.7, top-p 0.95) |
| `token_logprobs` | list[float] | log-probability of each token of the greedy answer |
| `nli_contra` | list[float] (10) | P(contradiction) between greedy answer and each sample |
| `bidir_entail` | 10x10 list of 0/1 | 1 if samples i and j entail each other (both directions) |
| `label_containment` | 0/1 | 1 = greedy answer correct (containment rule) |
| `label_judge` | 0/1 | 1 = correct according to the LLM judge (ambiguous counted as wrong) |

Check a file with `python scripts/validate_data.py`.
