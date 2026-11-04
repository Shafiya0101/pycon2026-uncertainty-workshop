"""Metacognitive Uncertainty Layer: interactive demo for the PyCon ES 2026 workshop.

Everything in tabs 1-4 is computed from the REAL study data (5 settings x 200 questions)
with the same code as the workshop notebooks (the bundled `uncertainty_fingerprint` package).
Tab 5 lets people paste their own answers and scores them live with DeBERTa NLI.
"""
import html
import inspect
import random

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import uncertainty_fingerprint as ufp

try:  # ZeroGPU on Hugging Face Spaces; harmless elsewhere
    import spaces
    gpu = spaces.GPU(duration=60)
except Exception:  # noqa: BLE001
    def gpu(fn):
        return fn

# ----------------------------------------------------------------------------- data
LABELS = {"mistral_triviaqa": "Mistral-7B · TriviaQA", "qwen2_7b_triviaqa": "Qwen2-7B · TriviaQA",
          "qwen2_1_5b_triviaqa": "Qwen2-1.5B · TriviaQA", "mistral_nq": "Mistral-7B · Natural Questions",
          "mistral_hotpotqa": "Mistral-7B · HotpotQA"}
NAME2KEY = {v: k for k, v in LABELS.items()}
SIG_NAMES = {"U_sc": "Self-consistency", "U_se": "Semantic entropy", "U_lp": "Log-probability", "U_nli": "NLI contradiction"}
SIG_HELP = {"U_sc": "do the 10 samples use the same words?", "U_se": "do they mean the same thing?",
            "U_lp": "how unsure was each token?", "U_nli": "do samples contradict the shown answer?"}
REGION_TEXT = {
    "Confident": ("#2c7a3f", "Samples agree in words and meaning."),
    "Gap-like": ("#c2410c", "Samples disagree: the model is guessing. Most wrong answers live here."),
    "Shallow": ("#8a6d1d", "Same words, different meanings. Rare."),
    "Surface-varying": ("#5b6474", "Different words, same meaning: paraphrasing, not doubt."),
}
DATA, DETECTORS = {}, {}
for key in ufp.SETTINGS:
    d = ufp.add_regions(ufp.add_signals(ufp.load_setting(key)))
    for c in SIG_NAMES:
        d[c + "_pct"] = (d[c].rank(method="min") - 1) / len(d)  # share of questions strictly less uncertain
    d["unanimous_wrong"] = (d.U_sc == 0) & (d.label_containment == 0)
    d["graders_disagree"] = d.label_containment != d.label_judge
    DATA[key] = d.reset_index(drop=True)
    DETECTORS[key] = ufp.Detector(score="Comb4").fit(ufp.load_setting(key), q_answer=0.45, q_abstain=0.80)
    DATA[key]["action"] = DETECTORS[key].predict(ufp.load_setting(key))["action"].to_numpy()

FILTERS = {
    "Any question": lambda d: d,
    "Confident fabrications (all 10 agree, still wrong)": lambda d: d[d.unanimous_wrong],
    "Gap-like (the model is guessing)": lambda d: d[d.region == "Gap-like"],
    "Wrong answers": lambda d: d[d.label_containment == 0],
    "Correct answers": lambda d: d[d.label_containment == 1],
    "The two graders disagree": lambda d: d[d.graders_disagree],
}

# ----------------------------------------------------------------------------- rendering
E = html.escape
CARD = "border:1px solid var(--border-color-primary);border-radius:12px;padding:16px 18px;margin-bottom:12px;background:var(--block-background-fill)"
LABEL = "font-size:12px;letter-spacing:1px;text-transform:uppercase;font-weight:700;opacity:.65;margin-bottom:6px"


def badge(text, color):
    return f'<span style="display:inline-block;background:{color};color:white;font-weight:700;padding:4px 12px;border-radius:999px;font-size:14px">{E(text)}</span>'


def signal_rows(r):
    out = []
    for c, name in SIG_NAMES.items():
        p = r[c + "_pct"]
        col = "#2c7a3f" if p < 0.4 else ("#b7791f" if p < 0.7 else "#c2410c")
        out.append(f"""<div style="margin:8px 0">
          <div style="display:flex;justify-content:space-between;font-size:14px"><span><b>{name}</b> <span style="opacity:.65">· {SIG_HELP[c]}</span></span>
          <span style="color:{col};font-weight:700">{r[c]:.3f}</span></div>
          <div style="background:var(--border-color-primary);border-radius:6px;height:10px;overflow:hidden;margin-top:4px">
          <div style="width:{max(p * 100, 2):.0f}%;height:100%;background:{col}"></div></div>
          <div style="font-size:12px;opacity:.65;margin-top:2px">more uncertain than {p * 100:.0f}% of questions in this setting</div></div>""")
    return "".join(out)


def samples_block(r):
    groups = {}
    for s, cid, con in zip(r["samples"], r["cluster_ids"], r["nli_contra"]):
        groups.setdefault(cid, []).append((s, con))
    parts = []
    for i, (cid, items) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        lis = "".join(f'<li style="margin:3px 0">{E(s)} <span style="font-size:12px;opacity:.6">· contradicts shown answer: {con:.0%}</span></li>' for s, con in items)
        parts.append(f'<div style="margin-bottom:8px"><b>Meaning group {i + 1}</b> · {len(items)} of {len(r["samples"])} samples<ul style="margin:4px 0 0 18px">{lis}</ul></div>')
    return "".join(parts)


def verdict_html(r):
    col, txt = REGION_TEXT[r["region"]]
    act = {"answer": ("Answer", "#2c7a3f"), "verify": ("Verify first", "#2563eb"), "abstain": ("Abstain", "#c2410c")}[r["action"]]
    return f"""<div style="{CARD}"><div style="{LABEL}">What the detector says (no answer key used)</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">Fingerprint {badge(r['region'], col)} &nbsp; Policy {badge(act[0], act[1])}</div>
      <div style="margin-top:8px;opacity:.85">{E(txt)}</div></div>"""


def truth_html(r):
    ok_c, ok_j = bool(r["label_containment"]), bool(r["label_judge"])
    gold = ", ".join(r["gold"][:4]) + (" …" if len(r["gold"]) > 4 else "")
    mark = lambda ok: ("✅ correct" if ok else "❌ wrong")
    extra = ""
    if r["unanimous_wrong"]:
        extra = '<div style="margin-top:8px;color:#c2410c;font-weight:700">⚠ Confident fabrication: all 10 samples agreed, and it is still wrong.</div>'
    elif ok_c != ok_j:
        extra = '<div style="margin-top:8px;color:#2563eb;font-weight:700">The two graders disagree on this one. Who is right?</div>'
    return f"""<div style="{CARD}"><div style="{LABEL}">The truth (answer key)</div>
      <div><b>Accepted answers:</b> {E(gold)}</div>
      <div style="margin-top:4px"><b>Word-matching rule:</b> {mark(ok_c)} &nbsp;·&nbsp; <b>AI judge:</b> {mark(ok_j)}</div>{extra}</div>"""


def question_html(r, setting):
    return f"""<div style="{CARD}"><div style="{LABEL}">{E(LABELS[setting])} · {E(r['qid'])}</div>
      <div style="font-size:20px;font-weight:700;margin-bottom:10px">{E(r['question'])}</div>
      <div style="{LABEL}">The AI's answer</div><div style="font-size:18px">{E(r['greedy'])}</div></div>"""


def full_html(r, setting):
    return (question_html(r, setting) + verdict_html(r) + truth_html(r)
            + f'<div style="{CARD}"><div style="{LABEL}">The four signals</div>{signal_rows(r)}</div>'
            + f'<details style="{CARD}"><summary style="cursor:pointer;font-weight:700">Show the 10 sampled answers, grouped by meaning</summary><div style="margin-top:10px">{samples_block(r)}</div></details>')


# ----------------------------------------------------------------------------- tab 1: game
def new_round(setting_label, kind, state):
    key = NAME2KEY[setting_label]
    pool = FILTERS[kind](DATA[key])
    if pool.empty:
        return state, "<p>No question of that kind in this setting. Try another.</p>", "", gr.update(interactive=False), gr.update(interactive=False)
    i = int(random.choice(pool.index))
    state = dict(state, key=key, i=i, answered=False)
    return state, question_html(DATA[key].iloc[i], key), "<p style='opacity:.7'>Make your call: would you show this answer to a user?</p>", gr.update(interactive=True), gr.update(interactive=True)


def guess(trust, state):
    if not state.get("key") or state.get("answered"):
        return state, gr.update(), gr.update(), gr.update(), gr.update()
    r = DATA[state["key"]].iloc[state["i"]]
    correct = bool(r["label_containment"])
    det_trust = r["region"] != "Gap-like"  # Policy A: abstain only on Gap-like
    state = dict(state, answered=True, n=state.get("n", 0) + 1,
                 you=state.get("you", 0) + int(trust == correct), det=state.get("det", 0) + int(det_trust == correct))
    you_txt = "✅ You were right." if trust == correct else "❌ You were fooled."
    det_txt = "✅ the detector was right too" if det_trust == correct else "❌ the detector got it wrong"
    head = f'<div style="{CARD}"><div style="font-size:20px;font-weight:800">{you_txt} And {det_txt}.</div>' \
           f'<div style="margin-top:6px">Score after {state["n"]} questions: <b>you {state["you"]}</b> · <b>detector {state["det"]}</b>' \
           f' <span style="opacity:.65">(the detector trusts every answer that is not Gap-like)</span></div></div>'
    body = verdict_html(r) + truth_html(r) + f'<div style="{CARD}"><div style="{LABEL}">The four signals</div>{signal_rows(r)}</div>' \
        + f'<details style="{CARD}"><summary style="cursor:pointer;font-weight:700">Show the 10 sampled answers</summary><div style="margin-top:10px">{samples_block(r)}</div></details>'
    return state, gr.update(), head + body, gr.update(interactive=False), gr.update(interactive=False)


# ----------------------------------------------------------------------------- tab 2: explore
def list_questions(setting_label, kind, search):
    d = FILTERS[kind](DATA[NAME2KEY[setting_label]])
    if search.strip():
        d = d[d.question.str.contains(search.strip(), case=False, regex=False)]
    choices = [f"{r.qid} · {r.question[:90]}" for r in d.itertuples()]
    return gr.update(choices=choices, value=choices[0] if choices else None), f"{len(choices)} question(s)"


def show_question(setting_label, choice):
    if not choice:
        return "<p style='opacity:.7'>Pick a question.</p>"
    key = NAME2KEY[setting_label]
    qid = choice.split(" · ")[0]
    r = DATA[key][DATA[key].qid == qid].iloc[0]
    return full_html(r, key)


# ----------------------------------------------------------------------------- tab 3: policy
def policy(setting_label, score, q_answer, q_abstain, verify_cost, verify_acc, label_col):
    key = NAME2KEY[setting_label]
    d = DATA[key]
    if q_answer > q_abstain:
        q_answer = q_abstain
    s = d[score].to_numpy()
    t1 = np.quantile(s, q_answer)
    t2 = np.inf if q_abstain >= 1 else np.quantile(s, q_abstain)
    acts = ufp.three_action(s, t1, t2)
    y = d[label_col].to_numpy()
    res = ufp.evaluate_actions(acts, y, verify_cost=verify_cost, verify_accuracy=verify_acc)
    base = y.mean()
    counts = pd.Series(acts).value_counts().reindex(["answer", "verify", "abstain"]).fillna(0).astype(int)
    ok = res["coverage"] >= 0.60
    summary = f"""<div style="{CARD}">
      <div style="display:flex;gap:28px;flex-wrap:wrap">
        <div><div style="{LABEL}">Accuracy</div><div style="font-size:32px;font-weight:800">{res['accuracy']:.1%}</div><div style="opacity:.7">answer everything: {base:.1%}</div></div>
        <div><div style="{LABEL}">Coverage</div><div style="font-size:32px;font-weight:800">{res['coverage']:.0%}</div><div style="opacity:.7">share of questions answered</div></div>
        <div><div style="{LABEL}">Cost / question</div><div style="font-size:32px;font-weight:800">{res['cost_per_question']:.2f}</div><div style="opacity:.7">toy model, see sliders</div></div>
      </div>
      <div style="margin-top:10px">Answer <b>{counts['answer']}</b> · Verify <b>{counts['verify']}</b> · Abstain <b>{counts['abstain']}</b> (of {len(d)})</div>
      <div style="margin-top:8px;font-weight:700;color:{'#2c7a3f' if ok else '#c2410c'}">{'🏆 Eligible for the leaderboard (coverage ≥ 60%)' if ok else 'Coverage below 60%: not eligible for the leaderboard'}</div></div>"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for sc_name, ls in zip(["U_sc", "U_lp", "Comb4"], ["--", ":", "-"]):
        cov, acc = ufp.risk_coverage_curve(d[sc_name], y)
        ax.plot(cov, acc, ls=ls, lw=2, label=f"answer safest first ({sc_name})")
    a_only = (acts == "answer")
    if a_only.any():
        ax.scatter([a_only.mean()], [y[a_only].mean()], s=140, color="black", zorder=5, label="your 'answer' zone alone")
    ax.scatter([res["coverage"]], [res["accuracy"]], s=180, marker="*", color="#c2410c", zorder=6, label="your policy (with verify)")
    ax.set_xlabel("Coverage (share answered)"); ax.set_ylabel("Accuracy on answered"); ax.set_ylim(max(0, base - 0.15), 1.01)
    ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower left"); fig.tight_layout()
    return summary, fig


def fair_race(setting_label, policy_name, label_col):
    d = DATA[NAME2KEY[setting_label]]
    out = ufp.matched_coverage_comparison(d, policy=policy_name[0], label_col=label_col, n_rep=200)
    out = out.reset_index().round(3)
    clear = out.ci_low.gt(0).any()
    msg = ("The fingerprint policy is clearly better than at least one simple score." if clear else
           "No interval is entirely above 0: at the same coverage, the fingerprint is not clearly better than a simple score.")
    return out, msg


# ----------------------------------------------------------------------------- tab 4: map
def fp_map(setting_label):
    key = NAME2KEY[setting_label]
    d = DATA[key]
    tau_sc, tau_se = ufp.fit_medians(d)
    rng = np.random.default_rng(0)
    jx, jy = rng.normal(0, .012, len(d)), rng.normal(0, .025, len(d))
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for lab, mk, col, name in [(1, "o", "#2c7a3f", "correct"), (0, "x", "#c2410c", "wrong")]:
        m = d.label_containment == lab
        ax.scatter(d.U_sc[m] + jx[m], d.U_se[m] + jy[m], marker=mk, c=col, alpha=.6, s=36, label=name)
    ax.axvline(tau_sc, color="black", ls="--"); ax.axhline(tau_se, color="black", ls="--")
    xm, ym = d.U_sc.max(), d.U_se.max()
    for name, (x, y) in {"Confident": (0.02, 0.03), "Gap-like": (0.98, 0.97), "Shallow": (0.02, 0.97), "Surface-varying": (0.98, 0.03)}.items():
        ax.text(x, y, name, transform=ax.transAxes, ha="left" if x < .5 else "right", va="bottom" if y < .5 else "top", fontsize=12, fontweight="bold")
    ax.set_xlabel("Words disagree (self-consistency uncertainty) →"); ax.set_ylabel("Meanings disagree (semantic entropy) →")
    ax.legend(loc="center right"); ax.grid(alpha=.25); ax.set_title(f"{LABELS[key]} · dashed lines = medians (no labels used)")
    fig.tight_layout()
    summ = ufp.region_summary(d).reset_index()
    summ["error_rate"] = (summ["error_rate"] * 100).round(1).astype(str) + "%"
    summ.columns = ["Region", "Questions", "Wrong"]
    return fig, summ


# ----------------------------------------------------------------------------- tab 5: your own answers
try:  # load DeBERTa once at start-up (ZeroGPU moves it to the GPU inside @gpu calls)
    from uncertainty_fingerprint.nli import NLIScorer
    _NLI = NLIScorer()
except Exception as _e:  # noqa: BLE001  e.g. no torch locally: tab 5 falls back to exact wording
    _NLI, _NLI_ERR = None, _e


def _get_nli():
    if _NLI is None:
        raise RuntimeError(f"NLI model unavailable: {_NLI_ERR!r}")
    return _NLI


@gpu
def _nli_annotate(rec):
    return _get_nli().annotate(rec)


def own_answers(question, shown, answers_text, calib_label):
    answers = [a.strip() for a in answers_text.splitlines() if a.strip()]
    if len(answers) < 3:
        return "<p>Paste at least 3 answers, one per line.</p>"
    shown = shown.strip() or answers[0]
    rec = {"question": question, "greedy": shown, "samples": answers}
    note = ""
    try:
        rec = _nli_annotate(rec)
    except Exception as e:  # noqa: BLE001  model unavailable: fall back to exact-wording groups
        rec["bidir_entail"] = [[int(ufp.normalize(a) == ufp.normalize(b)) for b in answers] for a in answers]
        rec["nli_contra"] = [0.0 if ufp.normalize(a) == ufp.normalize(shown) else 1.0 for a in answers]
        note = f"<p style='color:#b7791f'>NLI model not available ({E(type(e).__name__)}): meaning groups use exact wording only.</p>"
    cids = ufp.greedy_cluster(rec["bidir_entail"])
    sc, se = ufp.self_consistency_uncertainty(answers), ufp.semantic_entropy(cids)
    nli = ufp.nli_uncertainty(rec["nli_contra"])
    key = NAME2KEY[calib_label]
    tau_sc, tau_se = ufp.fit_medians(DATA[key])
    region = ufp.assign_region(sc, se, tau_sc, tau_se)
    col, txt = REGION_TEXT[region]
    groups = {}
    for a, c, con in zip(answers, cids, rec["nli_contra"]):
        groups.setdefault(c, []).append((a, con))
    gl = "".join(f'<div style="margin-bottom:6px"><b>Meaning group {i + 1}</b> · {len(v)} of {len(answers)}<ul style="margin:4px 0 0 18px">'
                 + "".join(f"<li>{E(a)} <span style='font-size:12px;opacity:.6'>· contradicts shown answer: {con:.0%}</span></li>" for a, con in v) + "</ul></div>"
                 for i, (c, v) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))))
    pct = lambda col_, v: (DATA[key][col_] < v).mean() * 100
    return f"""{note}<div style="{CARD}"><div style="{LABEL}">Fingerprint (medians from {E(calib_label)})</div>
      {badge(region, col)}<div style="margin-top:8px">{E(txt)}</div>
      <div style="margin-top:10px">Self-consistency <b>{sc:.2f}</b> (more uncertain than {pct('U_sc', sc):.0f}% of study answers) ·
      Semantic entropy <b>{se:.2f}</b> ({pct('U_se', se):.0f}%) · NLI contradiction <b>{nli:.2f}</b> ({pct('U_nli', nli):.0f}%)</div>
      <div style="margin-top:8px;opacity:.75">No log-probabilities from a chat app, so LP is skipped. Remember: agreement makes an answer more likely right, not guaranteed.</div></div>
      <div style="{CARD}"><div style="{LABEL}">Your answers, grouped by meaning</div>{gl}</div>"""


# ----------------------------------------------------------------------------- UI
CSS = ".gradio-container{max-width:1200px !important}"
SETTING_CHOICES = list(LABELS.values())

with gr.Blocks(title="Metacognitive Uncertainty Layer") as demo:
    gr.Markdown("# Metacognitive Uncertainty Layer\n**Can you tell when an AI is making things up?** "
                "Real answers from 3 models on 3 datasets (1,000 questions), scored with the four uncertainty signals from the PyCon ES 2026 workshop.")

    with gr.Tab("🎯 Trust or not?"):
        gr.Markdown("Read the question and the AI's answer. **Would you show it to a user?** Then see what the detector says, and the truth.")
        state = gr.State({})
        with gr.Row():
            g_set = gr.Dropdown(SETTING_CHOICES, value=SETTING_CHOICES[0], label="Model · dataset")
            g_kind = gr.Dropdown(list(FILTERS), value="Any question", label="Question type")
            g_new = gr.Button("🎲 New question", variant="primary")
        g_q = gr.HTML()
        with gr.Row():
            g_yes = gr.Button("✅ I'd trust it", interactive=False)
            g_no = gr.Button("❌ I wouldn't", interactive=False)
        g_out = gr.HTML()
        g_new.click(new_round, [g_set, g_kind, state], [state, g_q, g_out, g_yes, g_no])
        g_yes.click(lambda s: guess(True, s), state, [state, g_q, g_out, g_yes, g_no])
        g_no.click(lambda s: guess(False, s), state, [state, g_q, g_out, g_yes, g_no])

    with gr.Tab("🔍 Explore"):
        with gr.Row():
            e_set = gr.Dropdown(SETTING_CHOICES, value=SETTING_CHOICES[0], label="Model · dataset")
            e_kind = gr.Dropdown(list(FILTERS), value="Confident fabrications (all 10 agree, still wrong)", label="Show")
            e_search = gr.Textbox(label="Search questions", placeholder="e.g. Vinegar Joe")
        e_count = gr.Markdown()
        e_pick = gr.Dropdown([], label="Question")
        e_out = gr.HTML()
        for comp in (e_set, e_kind, e_search):
            comp.change(list_questions, [e_set, e_kind, e_search], [e_pick, e_count])
        e_pick.change(show_question, [e_set, e_pick], e_out)
        demo.load(list_questions, [e_set, e_kind, e_search], [e_pick, e_count])

    with gr.Tab("🎛️ Policy playground"):
        gr.Markdown("Two cut-offs on an uncertainty score: below the first **answer**, between them **verify**, above the second **abstain**. "
                    "Leaderboard rule: highest accuracy with coverage ≥ 60%.")
        with gr.Row():
            p_set = gr.Dropdown(SETTING_CHOICES, value=SETTING_CHOICES[0], label="Model · dataset")
            p_score = gr.Dropdown(["Comb4", "Comb3", "U_sc", "U_se", "U_lp", "U_nli"], value="Comb4", label="Uncertainty score")
            p_lab = gr.Radio(["label_containment", "label_judge"], value="label_containment", label="Grader")
        with gr.Row():
            p_qa = gr.Slider(0, 1, value=0.45, step=0.05, label="Answer below this quantile")
            p_qb = gr.Slider(0, 1, value=0.80, step=0.05, label="Abstain above this quantile (1 = never)")
        with gr.Row():
            p_cost = gr.Slider(0, 5, value=2.0, step=0.5, label="Extra cost of 'verify' (toy)")
            p_vacc = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Verify fixes this share of answers (toy)")
        p_sum = gr.HTML()
        p_plot = gr.Plot()
        inputs = [p_set, p_score, p_qa, p_qb, p_cost, p_vacc, p_lab]
        for comp in (p_set, p_score, p_lab):
            comp.change(policy, inputs, [p_sum, p_plot])
        for comp in (p_qa, p_qb, p_cost, p_vacc):  # update when the slider is released, not on every pixel
            comp.release(policy, inputs, [p_sum, p_plot])
        demo.load(policy, inputs, [p_sum, p_plot])
        gr.Markdown("### The fair race\nDoes the fingerprint policy beat a single score that answers **the same number** of questions? (200 resamples)")
        with gr.Row():
            r_pol = gr.Radio(["A · abstain on Gap-like", "B · answer only Confident"], value="A · abstain on Gap-like", label="Fingerprint policy")
            r_btn = gr.Button("Run the race")
        r_msg = gr.Markdown()
        r_tab = gr.Dataframe(interactive=False)
        r_btn.click(fair_race, [p_set, r_pol, p_lab], [r_tab, r_msg])

    with gr.Tab("🗺️ Fingerprint map"):
        gr.Markdown("Every dot is one answer. Green = correct, orange × = wrong. Look for the orange crosses in the **Confident** corner: those are confident fabrications.")
        m_set = gr.Dropdown(SETTING_CHOICES, value=SETTING_CHOICES[0], label="Model · dataset")
        m_plot = gr.Plot()
        m_tab = gr.Dataframe(interactive=False)
        m_set.change(fp_map, m_set, [m_plot, m_tab])
        demo.load(fp_map, m_set, [m_plot, m_tab])

    with gr.Tab("🧪 Your own answers"):
        gr.Markdown("Ask **any chatbot** the same question 5–10 times (a new chat each time) and paste the answers, one per line. "
                    "We group them by meaning with an NLI model and place them on the fingerprint map.")
        o_q = gr.Textbox(label="Your question", placeholder="Who was the first woman to win the Nobel Prize in Literature?")
        o_shown = gr.Textbox(label="The answer a user would see (optional, default: first line)")
        o_ans = gr.Textbox(label="Answers, one per line", lines=8)
        o_cal = gr.Dropdown(SETTING_CHOICES, value=SETTING_CHOICES[0], label="Compare with medians from")
        o_btn = gr.Button("Analyze my answers", variant="primary")
        o_out = gr.HTML()
        o_btn.click(own_answers, [o_q, o_shown, o_ans, o_cal], o_out)

    gr.Markdown("Thesis: *Metacognitive Uncertainty Estimation for Hallucination Detection in LLMs* · Shafiya Kausar, aivancity · "
                "[Workshop notebooks](https://github.com/Shafiya0101/pycon2026-uncertainty-workshop) · "
                "Finding: a fingerprint policy was never significantly better than a plain threshold at the same coverage; its value is explaining *why*.")

if __name__ == "__main__":
    kw = {"css": CSS} if "css" in inspect.signature(demo.launch).parameters else {}
    demo.launch(ssr_mode=False, **kw)
