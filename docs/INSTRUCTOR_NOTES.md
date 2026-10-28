# Instructor notes (not for attendees)

## Timetable (90 min, about 20 min of slides)
| Min | Block | Slides | Activity |
|---|---|---|---|
| 0-10 | Hook and setup | 3 | Challenge 1 poll, notebook 00 |
| 10-25 | Signals | 4 | Notebook 01 (Challenge 2) |
| 25-40 | Fingerprint | 3 | Notebook 02 (Challenge 3) |
| 40-54 | Policy | 3 | Notebook 03 (Challenge 4, leaderboard) |
| 54-64 | Results, pitfalls, demo | 6 | Failure cases, local Gradio |
| 64-78 | Break the detector | 0 | Notebook 04 (Challenge 5) |
| 78-84 | Sharing | 0 | One failure case per group |
| 84-86 | Contingency | 0 | |
| 86-90 | Takeaways, Q&A | 2 | Links again |

Cut order if late: bonus cells, Challenge 1 to one question, part of results block.

## Before the workshop checklist
- [ ] Export REAL data to `data/precomputed/` (see `scripts/export_precomputed_TEMPLATE.py`), run `python scripts/validate_data.py`
- [ ] Re-run `python tools/run_notebooks.py solutions` on the real data and re-read every printed number
- [ ] Check that the narrative still holds on real data (synthetic data is NOT the paper's data)
- [ ] Dry run on a clean machine (and a second OS) with someone at attendee level, with a stopwatch
- [ ] USB / zip copy of the repo, offline Gradio video + screenshots
- [ ] Never show an API key (none is needed)

## Rebuild notebooks after editing
Edit `tools/build_notebooks.py`, then `python tools/build_notebooks.py`. It writes both `notebooks/` and `solutions/`.

## Known gaps
* Colab path is untested (repo name in the first notebook cell must match the real GitHub repo).
* Gradio app is not included yet; use the thesis repo's `app.py` locally.
* Cost model in notebook 03 is a toy assumption (documented in `evaluate_actions`).
