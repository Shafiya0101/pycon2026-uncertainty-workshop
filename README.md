<p align="center">
<img src="docs/pycones2026-logo.png.png" alt="PyCon ES 2026 Barcelona" width="420">
</p>

# Detecting LLM Hallucinations with Uncertainty Fingerprinting: A Python Workshop

PyCon ES 2026 · Barcelona · 6 November 2026 · 90 minutes
Shafiya Kausar and Anuradha Kar, aivancity Paris

**Repository:** https://github.com/Shafiya0101/pycon2026-uncertainty-workshop
(Research code and thesis: https://github.com/Shafiya0101/metacognitive-uncertainty-layer)

**Live demo app:** https://pycon2026-uncertainty-workshop.streamlit.app/ (try it in your browser, nothing to install)

## What you need
* Python 3.10 or newer and a laptop. **No GPU, no API key, no model download, no internet during the session.**
* About 100 MB of disk space.

## Setup (do this before the workshop, 5 minutes)
```bash
git clone https://github.com/Shafiya0101/pycon2026-uncertainty-workshop.git
cd pycon2026-uncertainty-workshop
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```
Open `notebooks/00_setup_check.ipynb` and run both cells. You should see `✅ Setup OK`.

**No Python installed?** Open the notebook on Google Colab instead (needs internet):
[00_setup_check on Colab](https://colab.research.google.com/github/Shafiya0101/pycon2026-uncertainty-workshop/blob/main/notebooks/00_setup_check.ipynb)

## What you will do
| Notebook | Challenge | Time |
|---|---|---|
| `00_setup_check` | environment check | 2 min |
| `01_uncertainty_signals` | **2** build a self-consistency detector, then semantic entropy, log-probability and NLI | 12 min |
| `02_fingerprinting` | **3** compare the fingerprints of different models (run and read, no code to write) | 12 min |
| `03_adaptive_policy` | **4** explore an answer / verify / abstain policy (run, change two numbers, read) | 12 min |
| `04_break_the_detector` | **5** find where the detector fails | 15 min |

Challenge 1 ("Can you spot the hallucination?") is a poll on the slides.
Only notebook 01 has code to write. Stuck? Full solutions are in `solutions/` with the same file names
(for notebooks 00, 02, 03 and 04 the solution is identical to the notebook, because there is nothing to fill in).

## What we found (summary of the study)
We tested whether combining uncertainty signals, or reading their joint "fingerprint", beats a single signal.
On 3 models and 3 question-answering datasets (5 settings of 200 questions):
* Detection AUROC ranges from 0.63 to 0.82. Combinations were not significantly better than the SelfCheckGPT-NLI baseline.
* Self-consistency and semantic entropy are strongly correlated (Spearman 0.86 to 0.93).
* A fingerprint abstention policy was never significantly better than a plain threshold at the same coverage.
* Defining error types with correctness labels inflates results (a label-dependent policy looks like 93.9% accuracy at 81.5% coverage; the label-free version gets 91.2% at 56.5%).
* Unanimous agreement among 10 samples is not a guarantee: 5% to 29% of such Mistral-7B answers were wrong.

Limits: 200 questions per setting, 3 quantized models, short factual answers, labels are imperfect.

## Repository map
```
notebooks/        the workshop notebooks (TODO cells only in 01)
solutions/        completed notebooks (recovery points)
uncertainty_lib/  signals, fingerprint, policy, plotting, checks
data/             precomputed results (see data/README.md)
scripts/          export, validate and verify the data (plus synthetic placeholders)
tools/            build_notebooks.py (generates notebooks/ and solutions/) and run_notebooks.py (smoke test)
streamlit-app/    interactive demo (see streamlit-app/README.md)
slides/           slides (added after the workshop)
docs/             instructor notes
```

## Background
This workshop builds on the final-year thesis:
Kausar, S. (2026). *Intuition in AI Systems and Reducing Hallucination: Metacognitive Uncertainty
Estimation for Hallucination Detection in Large Language Models.* aivancity Paris, supervised by
Prof. Yasser Khalafaoui. DOI: [10.13140/RG.2.2.11958.66884](https://doi.org/10.13140/RG.2.2.11958.66884)
(preprint, CC BY 4.0). Research code: https://github.com/Shafiya0101/metacognitive-uncertainty-layer

**Note on results.** The thesis reports a fingerprint policy reaching 93.87% accuracy at 81.5%
coverage. That policy defines "knowledge gap" using correctness labels, so it cannot run in a real
system. A follow-up study with label-free fingerprints and matched-coverage comparisons finds
91.2% accuracy at 56.5% coverage, and plain thresholds do as well. The workshop teaches the
stricter evaluation.

## Sources and credits
* Datasets: TriviaQA (Joshi et al., 2017), Natural Questions (Kwiatkowski et al., 2019), HotpotQA (Yang et al., 2018).
* Models: Mistral-7B-Instruct-v0.3, Qwen2-7B-Instruct, Qwen2-1.5B-Instruct; NLI model DeBERTa-Large-MNLI.
* Methods: Self-consistency (Wang et al., 2023), Semantic entropy (Kuhn et al., 2023; Farquhar et al., 2024), SelfCheckGPT (Manakul et al., 2023).
* Answer correctness judge for the second labeling: Claude Haiku 5.5. Generative AI (Claude, Anthropic) assisted with code and text; the authors reviewed everything.

## Further reading
* Farquhar et al. (2024), *Detecting hallucinations in large language models using semantic entropy*, Nature 630.
* Manakul et al. (2023), *SelfCheckGPT*, EMNLP.
* Vashurin et al. (2025), *Benchmarking uncertainty quantification methods for LLMs with LM-Polygraph*, TACL.

License: MIT.

## Data and licenses

Code: MIT License (see LICENSE). The files in data/precomputed contain questions and reference answers from TriviaQA, Natural Questions and HotpotQA, plus model outputs. Those datasets keep their own licenses (Natural Questions and HotpotQA are released under CC BY-SA licenses, as far as we know), so please check the original sources before reusing the data.

## Troubleshooting

Windows PowerShell refuses to activate the environment (running scripts is disabled): use Command Prompt, or skip activation and call the environment's Python directly:

    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m jupyter lab

Windows says an Application Control policy blocked a package (pandas, for example): some company or university laptops block newly installed packages. Use the Google Colab links below, which need no installation.

Installation is slow or pip cannot find a package: check your internet connection, then use Colab.

Colab links (a Google account and internet are needed):

- [00_setup_check](https://colab.research.google.com/github/Shafiya0101/pycon2026-uncertainty-workshop/blob/main/notebooks/00_setup_check.ipynb)
- [01_uncertainty_signals](https://colab.research.google.com/github/Shafiya0101/pycon2026-uncertainty-workshop/blob/main/notebooks/01_uncertainty_signals.ipynb)
- [02_fingerprinting](https://colab.research.google.com/github/Shafiya0101/pycon2026-uncertainty-workshop/blob/main/notebooks/02_fingerprinting.ipynb)
- [03_adaptive_policy](https://colab.research.google.com/github/Shafiya0101/pycon2026-uncertainty-workshop/blob/main/notebooks/03_adaptive_policy.ipynb)
