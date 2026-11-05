"""Metacognitive Uncertainty Layer: lightweight Streamlit version.

Everything is computed from the REAL study data bundled in `uncertainty_fingerprint/data`
with the same code as the workshop notebooks. This version keeps 3 of the 5 settings and
does not load any neural network (no torch), so it fits Streamlit Community Cloud's memory.
"""
import html
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import uncertainty_fingerprint as ufp

st.set_page_config(page_title="Metacognitive Uncertainty Layer", page_icon="🧭", layout="wide")

# ----------------------------------------------------------------------------- settings
# To bring back more settings, add their keys here (the data files must be in
# uncertainty_fingerprint/data/). Original five: mistral_triviaqa, qwen2_7b_triviaqa,
# qwen2_1_5b_triviaqa, mistral_nq, mistral_hotpotqa.
ACTIVE = ["mistral_triviaqa", "qwen2_7b_triviaqa", "mistral_nq"]

ALL_LABELS = {"mistral_triviaqa": "Mistral-7B · TriviaQA", "qwen2_7b_triviaqa": "Qwen2-7B · TriviaQA",
              "qwen2_1_5b_triviaqa": "Qwen2-1.5B · TriviaQA", "mistral_nq": "Mistral-7B · Natural Questions",
              "mistral_hotpotqa": "Mistral-7B · HotpotQA"}
LABELS = {k: ALL_LABELS[k] for k in ACTIVE}
NAME2KEY = {v: k for k, v in LABELS.items()}
SETTING_CHOICES = list(LABELS.values())
SIG_NAMES = {"U_sc": "Self-consistency", "U_se": "Semantic entropy", "U_lp": "Log-probability", "U_nli": "NLI contradiction"}
SIG_HELP = {"U_sc": "do the 10 samples use the same words?", "U_se": "do they mean the same thing?",
            "U_lp": "how unsure was each token?", "U_nli": "do samples contradict the shown answer?"}
REGION_TEXT = {
    "Confident": ("#2c7a3f", "Samples agree in words and meaning."),
    "Gap-like": ("#c2410c", "Samples disagree: the model is guessing. Most wrong answers live here."),
    "Shallow": ("#8a6d1d", "Same words, different meanings. Rare."),
    "Surface-varying": ("#5b6474", "Different words, same meaning: paraphrasing, not doubt."),
}


@st.cache_resource(show_spinner="Loading study data…")
def load_data():
    data, detectors = {}, {}
    for key in ACTIVE:
        raw = ufp.load_setting(key)
        d = ufp.add_regions(ufp.add_signals(raw))
        for c in SIG_NAMES:
            d[c + "_pct"] = (d[c].rank(method="min") - 1) / len(d)  # share of questions strictly less uncertain
        d["unanimous_wrong"] = (d.U_sc == 0) & (d.label_containment == 0)
        d["graders_disagree"] = d.label_containment != d.label_judge
        d = d.reset_index(drop=True)
        det = ufp.Detector(score="Comb4").fit(raw, q_answer=0.45, q_abstain=0.80)
        d["action"] = det.predict(raw)["action"].to_numpy()
        data[key] = d
        detectors[key] = det
    return data, detectors


DATA, DETECTORS = load_data()

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
CARD = "border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:16px 18px;margin-bottom:12px;background:rgba(128,128,128,.07)"
LABEL = "font-size:12px;letter-spacing:1px;text-transform:uppercase;font-weight:700;opacity:.65;margin-bottom:6px"
TRACK = "rgba(128,128,128,.3)"


def show(markup: str):
    """Render HTML. Whitespace is collapsed so Markdown never mistakes indented HTML for a code block."""
    st.markdown(re.sub(r"\n\s*", "", markup), unsafe_allow_html=True)


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
          <div style="background:{TRACK};border-radius:6px;height:10px;overflow:hidden;margin-top:4px">
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
      <div>Fingerprint {badge(r['region'], col)} &nbsp; Policy {badge(act[0], act[1])}</div>
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
    return f"""<div style="{CARD}"><div style="{LABEL}">{E(LABELS[setting])} · {E(str(r['qid']))}</div>
      <div style="font-size:20px;font-weight:700;margin-bottom:10px">{E(r['question'])}</div>
      <div style="{LABEL}">The AI's answer</div><div style="font-size:18px">{E(r['greedy'])}</div></div>"""


def details_html(r):
    return (f'<div style="{CARD}"><div style="{LABEL}">The four signals</div>{signal_rows(r)}</div>'
            f'<details style="{CARD}"><summary style="cursor:pointer;font-weight:700">Show the 10 sampled answers, grouped by meaning</summary>'
            f'<div style="margin-top:10px">{samples_block(r)}</div></details>')


def full_html(r, setting):
    return question_html(r, setting) + verdict_html(r) + truth_html(r) + details_html(r)


# ----------------------------------------------------------------------------- tab 1: game
def new_round(setting_label, kind):
    key = NAME2KEY[setting_label]
    pool = FILTERS[kind](DATA[key])
    if pool.empty:
        st.session_state.game = {"empty": True}
        return
    st.session_state.game = {"key": key, "i": int(random.choice(pool.index)), "answered": False, "trust": None}


def guess(trust: bool):
    g = st.session_state.get("game", {})
    if not g.get("key") or g.get("answered"):
        return
    r = DATA[g["key"]].iloc[g["i"]]
    correct = bool(r["label_containment"])
    det_trust = r["region"] != "Gap-like"  # Policy A: abstain only on Gap-like
    sc = st.session_state.score
    sc["n"] += 1
    sc["you"] += int(trust == correct)
    sc["det"] += int(det_trust == correct)
    g["answered"], g["trust"], g["det_trust"] = True, trust, det_trust


def tab_game():
    st.markdown("Read the question and the AI's answer. **Would you show it to a user?** Then see what the detector says, and the truth.")
    st.session_state.setdefault("game", {})
    st.session_state.setdefault("score", {"n": 0, "you": 0, "det": 0})
    c1, c2, c3 = st.columns([2, 3, 1])
    g_set = c1.selectbox("Model · dataset", SETTING_CHOICES, key="g_set")
    g_kind = c2.selectbox("Question type", list(FILTERS), key="g_kind")
    c3.write("")
    c3.button("🎲 New question", type="primary", on_click=new_round, args=(g_set, g_kind), key="g_new", use_container_width=True)
    g = st.session_state.game
    if g.get("empty"):
        st.info("No question of that kind in this setting. Try another.")
        return
    if not g.get("key"):
        st.caption("Press **New question** to start.")
        return
    r = DATA[g["key"]].iloc[g["i"]]
    show(question_html(r, g["key"]))
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("✅ I'd trust it", on_click=guess, args=(True,), disabled=g["answered"], key="g_yes", use_container_width=True)
    b2.button("❌ I wouldn't", on_click=guess, args=(False,), disabled=g["answered"], key="g_no", use_container_width=True)
    if not g["answered"]:
        st.caption("Make your call: would you show this answer to a user?")
        return
    correct, sc = bool(r["label_containment"]), st.session_state.score
    you_txt = "✅ You were right." if g["trust"] == correct else "❌ You were fooled."
    det_txt = "✅ the detector was right too" if g["det_trust"] == correct else "❌ the detector got it wrong"
    show(f'<div style="{CARD}"><div style="font-size:20px;font-weight:800">{you_txt} And {det_txt}.</div>'
         f'<div style="margin-top:6px">Score after {sc["n"]} questions: <b>you {sc["you"]}</b> · <b>detector {sc["det"]}</b>'
         f' <span style="opacity:.65">(the detector trusts every answer that is not Gap-like)</span></div></div>')
    show(verdict_html(r) + truth_html(r) + details_html(r))


# ----------------------------------------------------------------------------- tab 2: explore
def tab_explore():
    c1, c2, c3 = st.columns([2, 3, 2])
    setting = c1.selectbox("Model · dataset", SETTING_CHOICES, key="e_set")
    kind = c2.selectbox("Show", list(FILTERS), index=1, key="e_kind")
    search = c3.text_input("Search questions", placeholder="e.g. Vinegar Joe", key="e_search")
    key = NAME2KEY[setting]
    d = FILTERS[kind](DATA[key])
    if search.strip():
        d = d[d.question.str.contains(search.strip(), case=False, regex=False)]
    st.markdown(f"**{len(d)} question(s)**")
    if d.empty:
        return
    ids = d.qid.tolist()
    names = dict(zip(d.qid, d.question))
    qid = st.selectbox("Question", ids, format_func=lambda q: f"{q} · {names[q][:90]}", key="e_pick")
    show(full_html(d[d.qid == qid].iloc[0], key))


# ----------------------------------------------------------------------------- tab 3: policy
def policy_view(key, score, q_answer, q_abstain, verify_cost, verify_acc, label_col):
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
    show(f"""<div style="{CARD}">
      <div style="display:flex;gap:28px;flex-wrap:wrap">
        <div><div style="{LABEL}">Accuracy</div><div style="font-size:32px;font-weight:800">{res['accuracy']:.1%}</div><div style="opacity:.7">answer everything: {base:.1%}</div></div>
        <div><div style="{LABEL}">Coverage</div><div style="font-size:32px;font-weight:800">{res['coverage']:.0%}</div><div style="opacity:.7">share of questions answered</div></div>
        <div><div style="{LABEL}">Cost / question</div><div style="font-size:32px;font-weight:800">{res['cost_per_question']:.2f}</div><div style="opacity:.7">toy model, see sliders</div></div>
      </div>
      <div style="margin-top:10px">Answer <b>{counts['answer']}</b> · Verify <b>{counts['verify']}</b> · Abstain <b>{counts['abstain']}</b> (of {len(d)})</div>
      <div style="margin-top:8px;font-weight:700;color:{'#2c7a3f' if ok else '#c2410c'}">{'🏆 Eligible for the leaderboard (coverage ≥ 60%)' if ok else 'Coverage below 60%: not eligible for the leaderboard'}</div></div>""")
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
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data(show_spinner="Running 200 resamples…")
def race_table(key, policy_letter, label_col):
    out = ufp.matched_coverage_comparison(DATA[key], policy=policy_letter, label_col=label_col, n_rep=200)
    return out.reset_index().round(3)


def tab_policy():
    st.markdown("Two cut-offs on an uncertainty score: below the first **answer**, between them **verify**, above the second **abstain**. "
                "Leaderboard rule: highest accuracy with coverage ≥ 60%.")
    c1, c2, c3 = st.columns(3)
    setting = c1.selectbox("Model · dataset", SETTING_CHOICES, key="p_set")
    score = c2.selectbox("Uncertainty score", ["Comb4", "Comb3", "U_sc", "U_se", "U_lp", "U_nli"], key="p_score")
    label_col = c3.radio("Grader", ["label_containment", "label_judge"], horizontal=True, key="p_lab")
    c1, c2 = st.columns(2)
    q_a = c1.slider("Answer below this quantile", 0.0, 1.0, 0.45, 0.05, key="p_qa")
    q_b = c2.slider("Abstain above this quantile (1 = never)", 0.0, 1.0, 0.80, 0.05, key="p_qb")
    c1, c2 = st.columns(2)
    cost = c1.slider("Extra cost of 'verify' (toy)", 0.0, 5.0, 2.0, 0.5, key="p_cost")
    vacc = c2.slider("Verify fixes this share of answers (toy)", 0.5, 1.0, 0.9, 0.05, key="p_vacc")
    key = NAME2KEY[setting]
    policy_view(key, score, q_a, q_b, cost, vacc, label_col)

    st.markdown("### The fair race")
    st.markdown("Does the fingerprint policy beat a single score that answers **the same number** of questions? (200 resamples)")
    pol = st.radio("Fingerprint policy", ["A · abstain on Gap-like", "B · answer only Confident"], horizontal=True, key="r_pol")
    if st.button("Run the race", key="r_btn"):
        out = race_table(key, pol[0], label_col)
        clear = out.ci_low.gt(0).any()
        st.markdown("The fingerprint policy is clearly better than at least one simple score." if clear else
                    "No interval is entirely above 0: at the same coverage, the fingerprint is not clearly better than a simple score.")
        st.dataframe(out, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------- tab 4: map
def tab_map():
    st.markdown("Every dot is one answer. Green = correct, orange × = wrong. Look for the orange crosses in the **Confident** corner: those are confident fabrications.")
    setting = st.selectbox("Model · dataset", SETTING_CHOICES, key="m_set")
    key = NAME2KEY[setting]
    d = DATA[key]
    tau_sc, tau_se = ufp.fit_medians(d)
    rng = np.random.default_rng(0)
    jx, jy = rng.normal(0, .012, len(d)), rng.normal(0, .025, len(d))
    fig, ax = plt.subplots(figsize=(7, 5.2))
    for lab, mk, col, name in [(1, "o", "#2c7a3f", "correct"), (0, "x", "#c2410c", "wrong")]:
        m = d.label_containment == lab
        ax.scatter(d.U_sc[m] + jx[m], d.U_se[m] + jy[m], marker=mk, c=col, alpha=.6, s=36, label=name)
    ax.axvline(tau_sc, color="black", ls="--"); ax.axhline(tau_se, color="black", ls="--")
    for name, (x, y) in {"Confident": (0.02, 0.03), "Gap-like": (0.98, 0.97), "Shallow": (0.02, 0.97), "Surface-varying": (0.98, 0.03)}.items():
        ax.text(x, y, name, transform=ax.transAxes, ha="left" if x < .5 else "right", va="bottom" if y < .5 else "top", fontsize=12, fontweight="bold")
    ax.set_xlabel("Words disagree (self-consistency uncertainty) →"); ax.set_ylabel("Meanings disagree (semantic entropy) →")
    ax.legend(loc="center right"); ax.grid(alpha=.25); ax.set_title(f"{LABELS[key]} · dashed lines = medians (no labels used)")
    fig.tight_layout()
    left, right = st.columns([3, 2])
    left.pyplot(fig)
    plt.close(fig)
    summ = ufp.region_summary(d).reset_index()
    summ["error_rate"] = (summ["error_rate"] * 100).round(1).astype(str) + "%"
    summ.columns = ["Region", "Questions", "Wrong"]
    right.dataframe(summ, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------- tab 5: break the detector
CAUSES = {
    "fabrication": ("🤖 Fabrication", "The model really is wrong.", "Says “Sydney” for the capital of Australia."),
    "strict_gold": ("📏 Answer key too strict", "Right answer, but the key wants exact words.", "Says “physician”; the key says “doctor”."),
    "label_noise": ("🏷️ Grading mistake", "The label itself is wrong.", "A wrong answer gets credit for mentioning the right word."),
    "ambiguous": ("🤷 Ambiguous", "Reasonable people would disagree.", "“Who invented the telephone?” Bell or Meucci?"),
}


@st.cache_resource(show_spinner="Loading all five study settings…")
def load_break_data():
    """Break the detector always uses all five settings (like notebook 04), whatever ACTIVE is."""
    frames = [ufp.add_regions(ufp.add_signals(ufp.load_setting(k))) for k in ufp.SETTINGS]
    return pd.concat(frames, ignore_index=True)


def save_verdict(case_id, widget_key):
    st.session_state.setdefault("verdicts", {})[case_id] = st.session_state[widget_key]


def tab_break():
    allq = load_break_data()
    st.markdown("### Break the detector")
    st.markdown("The detector said “fine”, but the answer was marked wrong. **Whose fault is it?** "
                "Find answers where the detector is wrong or overconfident, and say *why*. Same data as notebook 04, all five settings.")
    st.session_state.setdefault("verdicts", {})

    # Step 1
    st.markdown("#### Step 1 · How often is *unanimous* still wrong?")
    st.caption("Unanimous = all 10 samples identical. If agreement guaranteed correctness, the last column would be 0%.")
    g = allq.groupby("setting")
    tab = pd.DataFrame({"Questions": g.size(),
                        "Unanimous": g.apply(lambda d: int((d.U_sc == 0).sum())),
                        "Unanimous and wrong": g.apply(lambda d: int(((d.U_sc == 0) & (d.label_containment == 0)).sum()))})
    tab.loc["All five settings"] = tab.sum()
    tab["Share wrong"] = (tab["Unanimous and wrong"] / tab["Unanimous"].replace(0, np.nan)).map(lambda v: "-" if pd.isna(v) else f"{v:.1%}")
    tab.index = [ALL_LABELS.get(i, i) for i in tab.index]
    st.dataframe(tab, use_container_width=True)
    total = int(tab.loc["All five settings", "Unanimous and wrong"])
    st.markdown(f"**{total} confident fabrications** across the five settings: the model gave the same wrong answer all 10 times.")

    # Step 2
    st.markdown("#### Step 2 · Judge the confident-but-wrong cases")
    cols = st.columns(4)
    for col, (code, (title, what, ex)) in zip(cols, CAUSES.items()):
        col.markdown(f"**{title}**  \n{what}  \n*{ex}*")
    cases = allq[(allq.U_sc == 0) & (allq.label_containment == 0)].reset_index(drop=True)
    cases["case_id"] = cases.setting + " · " + cases.qid.astype(str)
    names = dict(zip(cases.case_id, cases.question))
    verdicts = st.session_state.verdicts
    pick = st.selectbox("Case", cases.case_id.tolist(), key="b_case",
                        format_func=lambda c: ("✔ " if c in verdicts else "") + f"{c} · {names[c][:70]}")
    r = cases[cases.case_id == pick].iloc[0]
    gold = ", ".join(r["gold"][:4]) + (" …" if len(r["gold"]) > 4 else "")
    show(f"""<div style="{CARD}"><div style="{LABEL}">{E(ALL_LABELS[r['setting']])} · {E(str(r['qid']))}</div>
      <div style="font-size:20px;font-weight:700;margin-bottom:10px">{E(r['question'])}</div>
      <div style="{LABEL}">The AI's answer (all 10 samples agree)</div><div style="font-size:18px">{E(r['greedy'])}</div>
      <div style="margin-top:10px"><b>Accepted answers:</b> {E(gold)}</div>
      <div style="margin-top:4px"><b>Word-matching rule:</b> ❌ wrong &nbsp;·&nbsp; <b>AI judge:</b> {'✅ correct' if r['label_judge'] else '❌ wrong'}</div></div>""")
    wkey = "b_verdict_" + pick
    codes = list(CAUSES)
    st.radio("Your verdict", codes, index=codes.index(verdicts[pick]) if pick in verdicts else None,
             format_func=lambda c: CAUSES[c][0] + " · " + CAUSES[c][1], key=wkey, on_change=save_verdict, args=(pick, wkey))
    if verdicts:
        tally = pd.Series(verdicts).value_counts().reindex(codes).fillna(0).astype(int)
        t1, t2 = st.columns([2, 3])
        t1.markdown(f"**Your verdicts so far: {len(verdicts)} of {len(cases)}**")
        t1.dataframe(tally.rename("cases").to_frame().rename(index=lambda c: CAUSES[c][0]), use_container_width=True)
        out = pd.DataFrame({"case": list(verdicts), "verdict": list(verdicts.values())})
        t2.download_button("⬇ Download my verdicts (CSV)", out.to_csv(index=False).encode(), "my_verdicts.csv", "text/csv", key="b_dl")
        t2.caption("Verdicts live in this browser tab only. Download them before you close it.")

    # Step 3
    st.markdown("#### Step 3 · Where do the two graders disagree?")
    dis = allq[allq.label_containment != allq.label_judge]
    st.markdown(f"**{len(dis)} of {len(allq)} answers ({len(dis) / len(allq):.1%})** get different labels from the word-matching rule and the AI judge. "
                "Disagreements show how label noise moves results.")
    st.dataframe(dis[["setting", "question", "greedy", "gold", "label_containment", "label_judge"]].head(8)
                 .assign(gold=lambda d: d.gold.map(lambda x: ", ".join(x[:3]))), hide_index=True, use_container_width=True)

    # Step 4
    st.markdown("#### Step 4 · The detector's most confident mistakes")
    st.caption("Wrong answers with the lowest combined uncertainty (Comb4): the system would answer these without hesitation.")
    worst = allq[allq.label_containment == 0].nsmallest(10, "Comb4")
    st.dataframe(worst[["setting", "question", "greedy", "gold", "U_sc", "U_se", "U_lp", "U_nli", "Comb4"]]
                 .assign(gold=lambda d: d.gold.map(lambda x: ", ".join(x[:3]))).round(2), hide_index=True, use_container_width=True)

    st.markdown("#### Discussion")
    st.markdown("1. Which failures could a **better detector** fix, and which need **outside evidence** (retrieval, a database, a human)?\n"
                "2. How much of what looks like “model error” is actually **label error**?\n"
                "3. Your product answers medical or legal questions. Which **coverage** would you accept, and who handles the abstentions?\n"
                "4. Would you add a fifth sampling signal, or something of a different kind?")


# ----------------------------------------------------------------------------- tab 6: your own answers (lightweight)
def tab_own():
    st.markdown("Ask **any chatbot** the same question 5–10 times (a new chat each time) and paste the answers, one per line.")
    st.info("Lightweight mode: this hosted version has no NLI model, so answers are grouped by **exact wording** after normalisation "
            "(case, punctuation, articles). Paraphrases count as different meanings, so semantic entropy and NLI contradiction are cruder than in the full demo.")
    question = st.text_input("Your question", placeholder="Who was the first woman to win the Nobel Prize in Literature?", key="o_q")
    shown = st.text_input("The answer a user would see (optional, default: first line)", key="o_shown")
    answers_text = st.text_area("Answers, one per line", height=180, key="o_ans")
    calib = st.selectbox("Compare with medians from", SETTING_CHOICES, key="o_cal")
    if not st.button("Analyze my answers", type="primary", key="o_btn"):
        return
    answers = [a.strip() for a in answers_text.splitlines() if a.strip()]
    if len(answers) < 3:
        st.warning("Paste at least 3 answers, one per line.")
        return
    shown = shown.strip() or answers[0]
    bidir = [[int(ufp.normalize(a) == ufp.normalize(b)) for b in answers] for a in answers]
    contra = [0.0 if ufp.normalize(a) == ufp.normalize(shown) else 1.0 for a in answers]
    cids = ufp.greedy_cluster(bidir)
    sc, se = ufp.self_consistency_uncertainty(answers), ufp.semantic_entropy(cids)
    nli = ufp.nli_uncertainty(contra)
    key = NAME2KEY[calib]
    tau_sc, tau_se = ufp.fit_medians(DATA[key])
    region = ufp.assign_region(sc, se, tau_sc, tau_se)
    col, txt = REGION_TEXT[region]
    groups = {}
    for a, c, con in zip(answers, cids, contra):
        groups.setdefault(c, []).append((a, con))
    gl = "".join(f'<div style="margin-bottom:6px"><b>Group {i + 1}</b> · {len(v)} of {len(answers)}<ul style="margin:4px 0 0 18px">'
                 + "".join(f"<li>{E(a)} <span style='font-size:12px;opacity:.6'>· contradicts shown answer: {con:.0%}</span></li>" for a, con in v) + "</ul></div>"
                 for i, (c, v) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))))
    pct = lambda c_, v: (DATA[key][c_] < v).mean() * 100
    show(f"""<div style="{CARD}"><div style="{LABEL}">Fingerprint (medians from {E(calib)})</div>
      {badge(region, col)}<div style="margin-top:8px">{E(txt)}</div>
      <div style="margin-top:10px">Self-consistency <b>{sc:.2f}</b> (more uncertain than {pct('U_sc', sc):.0f}% of study answers) ·
      Semantic entropy <b>{se:.2f}</b> ({pct('U_se', se):.0f}%) · NLI contradiction <b>{nli:.2f}</b> ({pct('U_nli', nli):.0f}%)</div>
      <div style="margin-top:8px;opacity:.75">No log-probabilities from a chat app, so LP is skipped. Remember: agreement makes an answer more likely right, not guaranteed.</div></div>
      <div style="{CARD}"><div style="{LABEL}">Your answers, grouped by exact wording</div>{gl}</div>""")


# ----------------------------------------------------------------------------- UI
n_models = len({k.rsplit("_", 1)[0] for k in ACTIVE})
n_sets = len({k.rsplit("_", 1)[1] for k in ACTIVE})
n_q = sum(len(DATA[k]) for k in ACTIVE)
st.title("🧭 Metacognitive Uncertainty Layer")
st.markdown(f"**Can you tell when an AI is making things up?** Real answers from {n_models} models on {n_sets} datasets "
            f"({n_q:,} questions), scored with the four uncertainty signals from the PyCon ES 2026 workshop.")

t1, t2, t3, t4, t5, t6 = st.tabs(["🎯 Trust or not?", "🔍 Explore", "🎛️ Policy playground", "🗺️ Fingerprint map",
                                  "🕵️ Break the detector", "🧪 Your own answers"])
with t1:
    tab_game()
with t2:
    tab_explore()
with t3:
    tab_policy()
with t4:
    tab_map()
with t5:
    tab_break()
with t6:
    tab_own()

st.markdown("---")
st.markdown("Thesis: *Metacognitive Uncertainty Estimation for Hallucination Detection in LLMs* · Shafiya Kausar, aivancity · "
            "[Workshop notebooks](https://github.com/Shafiya0101/pycon2026-uncertainty-workshop) · "
            "Finding: a fingerprint policy was never significantly better than a plain threshold at the same coverage; its value is explaining *why*.")
