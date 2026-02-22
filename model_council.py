import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
import concurrent.futures
import os
import json
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Model Council",
    page_icon="⚖️",
    layout="wide"
)

# ─────────────────────────────────────────────
# History helpers
# ─────────────────────────────────────────────
HISTORY_DIR = Path("council_sessions")
HISTORY_DIR.mkdir(exist_ok=True)


def save_session(data):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data["timestamp"] = ts
    fp = HISTORY_DIR / f"{ts}.json"
    with open(fp, "w") as f:
        json.dump(data, f, indent=2)
    return ts


def load_all_sessions():
    out = []
    for fp in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            with open(fp) as f:
                s = json.load(f)
            s["_filename"] = fp.name
            out.append(s)
        except Exception:
            pass
    return out


def load_session(filename):
    with open(HISTORY_DIR / filename) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
if "view" not in st.session_state:
    st.session_state.view = "new"
if "running" not in st.session_state:
    st.session_state.running = False


# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────
def query_claude(prompt, api_key, model, max_tok):
    client = Anthropic(api_key=api_key)
    r = client.messages.create(
        model=model, max_tokens=max_tok,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text


def query_gpt(prompt, api_key, model, max_tok):
    client = OpenAI(api_key=api_key)
    r = client.chat.completions.create(
        model=model, max_tokens=max_tok,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


def query_grok(prompt, api_key, model, max_tok):
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    r = client.chat.completions.create(
        model=model, max_tokens=max_tok,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content


def run_parallel(task_dict):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fn): name for name, fn in task_dict.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                results[name] = f"⚠️ Error: {e}"
    return results


# ─────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────
def build_critique_prompt(q, my_r1, others_r1):
    others_text = "\n\n".join([f"=== {n}'s answer ===\n{a}" for n, a in others_r1.items()])
    return f"""You are participating in a structured debate with two other AI models. All three of you were asked the same question.

ORIGINAL QUESTION: {q}

YOUR INITIAL ANSWER:
{my_r1}

THE OTHER TWO MODELS ANSWERED AS FOLLOWS:
{others_text}

Now critically review all three answers including your own:
1. AGREE: What do the other models get right? Where does your view align with theirs?
2. DISAGREE: Where do you push back? Defend your position with specific reasoning.
3. LEARNED: Did the others raise angles or evidence you hadn't considered? Be honest.
4. REFINED ANSWER: Give your best updated answer incorporating the strongest thinking from this review.

Be intellectually rigorous. It is fine to change your view — and fine to defend it."""


def build_final_position_prompt(q, my_r1, others_r1, my_r2, others_r2):
    others_r1_text = "\n\n".join([f"=== {n} — Round 1 ===\n{a}" for n, a in others_r1.items()])
    others_r2_text = "\n\n".join([f"=== {n} — Round 2 ===\n{a}" for n, a in others_r2.items()])
    return f"""This is the FINAL ROUND of a structured three-round debate between three AI models.

ORIGINAL QUESTION: {q}

YOUR Round 1 answer:
{my_r1}

Other models' Round 1 answers:
{others_r1_text}

YOUR Round 2 critique:
{my_r2}

Other models' Round 2 critiques:
{others_r2_text}

You have now seen the full debate — both what the other models said initially and how they responded to criticism.

Give your DEFINITIVE FINAL POSITION:
1. FINAL ANSWER: Your settled, conclusive answer to the original question.
2. WHAT CHANGED: How did your view evolve (or not) across the three rounds? What moved you?
3. STRONGEST OPPOSING POINT: What is the most compelling argument against your position, and why does your view still hold?
4. ONE SENTENCE: The single most important insight from this entire debate that someone should walk away with."""


def build_synthesis_prompt(q, r1, r2, r3, synthesizer):
    def fmt(d):
        return "\n\n".join([f"=== {n} ===\n{a}" for n, a in d.items()])
    return f"""You are a neutral senior research analyst. Three AI models (Claude, ChatGPT, and Grok) have just completed a structured 3-round debate on a research question. Your job is to write a comprehensive synthesis report.

ORIGINAL QUESTION: {q}

━━━ ROUND 1 — Independent Initial Answers ━━━
{fmt(r1)}

━━━ ROUND 2 — Cross-Model Critique ━━━
{fmt(r2)}

━━━ ROUND 3 — Final Definitive Positions ━━━
{fmt(r3)}

Write a structured synthesis with these sections:

## CONSENSUS
What did all three models ultimately agree on by Round 3? How strong and meaningful is this consensus?

## KEY DIVERGENCES
Where did they genuinely disagree by the end? Describe the nature of each disagreement (factual, interpretive, emphasis, framing). Which position appears most defensible, and why?

## HOW VIEWS EVOLVED
How did each model's position shift across the three rounds? Who was most open to changing their view? What caused the changes?

## OPTIMAL ANSWER
The best, most complete answer to the original question — drawing from all three models' strongest contributions across all rounds.

## UNIQUE CONTRIBUTIONS
One key insight from each model (Claude, ChatGPT, Grok) that the others missed or underemphasized.

## WHAT THIS DEBATE REVEALS
What does the pattern of agreement, disagreement, and evolution tell us about the nature of this question — its complexity, uncertainty, or the current state of knowledge?

## CONFIDENCE RATING
High / Medium / Low alignment by the end, and what that means for how much to trust the answer.

Write as a genuine research synthesis — thorough, honest about uncertainty, and immediately useful."""


# ─────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────
def show_three_cols(data):
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**🟣 Claude**")
            st.markdown(data.get("Claude", "*No response*"))
    with c2:
        with st.container(border=True):
            st.markdown("**🟢 ChatGPT**")
            st.markdown(data.get("ChatGPT", "*No response*"))
    with c3:
        with st.container(border=True):
            st.markdown("**🔴 Grok**")
            st.markdown(data.get("Grok", "*No response*"))


def build_export(s):
    def sec(d):
        return "\n\n".join([f"### {n}\n{a}" for n, a in d.items()])
    return f"""# ⚖️ Model Council Session
**Date:** {s.get('timestamp', '')}
**Synthesizer:** {s.get('synthesizer', 'Claude')}

---

## Original Prompt
{s['prompt']}

---

## Round 1 — Initial Answers
{sec(s['round1'])}

---

## Round 2 — Cross-Model Critique
{sec(s['round2'])}

---

## Round 3 — Final Positions
{sec(s['round3'])}

---

## Final Synthesis (by {s.get('synthesizer', 'Claude')})
{s['synthesis']}
"""


def display_saved_session(s):
    st.caption(f"🕐 {s['timestamp'].replace('_', ' ')}")
    with st.container(border=True):
        st.markdown(f"**Prompt:** {s['prompt']}")

    st.subheader("🔵 Round 1 — Initial Answers")
    show_three_cols(s["round1"])

    st.subheader("🟠 Round 2 — Cross-Model Critique")
    show_three_cols(s["round2"])

    st.subheader("🔁 Round 3 — Final Positions")
    show_three_cols(s["round3"])

    st.subheader(f"⚡ Final Synthesis — by {s.get('synthesizer', 'Claude')}")
    with st.container(border=True):
        st.markdown(s["synthesis"])

    st.download_button(
        "📥 Download This Session",
        build_export(s),
        file_name=f"council_{s['timestamp']}.md",
        mime="text/markdown",
        use_container_width=True
    )


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Model Council")
    st.caption("3-round AI debate → research synthesis")

    if st.button("✏️  New Session", use_container_width=True, type="primary"):
        st.session_state.view = "new"
        st.rerun()

    st.divider()

    # History
    all_sessions = load_all_sessions()
    if all_sessions:
        st.subheader("🕐 Prior Sessions")
        for s in all_sessions:
            label = s["prompt"][:40] + "…" if len(s["prompt"]) > 40 else s["prompt"]
            date_display = s["timestamp"][:10]
            btn_label = f"{label}  \n*{date_display}*"
            if st.button(btn_label, key=s["_filename"], use_container_width=True):
                st.session_state.view = s["_filename"]
                st.rerun()
    else:
        st.caption("No prior sessions yet.\nRun your first query to see history here.")

    st.divider()

    with st.expander("🔑 API Keys", expanded=True):
        anthropic_key = st.text_input(
            "Anthropic (Claude)",
            value=os.getenv("ANTHROPIC_API_KEY", ""),
            type="password", placeholder="sk-ant-..."
        )
        openai_key = st.text_input(
            "OpenAI (ChatGPT)",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password", placeholder="sk-..."
        )
        grok_key = st.text_input(
            "xAI (Grok)",
            value=os.getenv("XAI_API_KEY", ""),
            type="password", placeholder="xai-..."
        )

    with st.expander("⚙️ Settings", expanded=False):
        claude_model = st.selectbox(
            "Claude model",
            ["claude-sonnet-4-6", "claude-opus-4-6"], index=0,
            help="Sonnet 4.6 released Feb 17 2026 · Opus 4.6 released Feb 5 2026"
        )
        gpt_model = st.selectbox(
            "ChatGPT model",
            ["gpt-5.2", "gpt-4o"], index=0,
            help="5.2 = latest via Chat API · 5.2-pro requires a different API · 4o = reliable fallback"
        )
        grok_model = st.selectbox(
            "Grok model",
            ["grok-4", "grok-3-beta"], index=0,
            help="Grok 4 = latest via API · Heavy & 4.20 not yet on public API"
        )
        synthesizer = st.selectbox(
            "Who writes the final synthesis?",
            ["Claude", "ChatGPT", "Grok"], index=0
        )
        max_tokens = st.slider("Max tokens per response", 500, 4000, 2000, 100)


# ─────────────────────────────────────────────
# MAIN AREA — History view
# ─────────────────────────────────────────────
if st.session_state.view != "new":
    try:
        session_data = load_session(st.session_state.view)
        display_saved_session(session_data)
    except Exception as e:
        st.error(f"Could not load session: {e}")
        st.session_state.view = "new"
        st.rerun()

# ─────────────────────────────────────────────
# MAIN AREA — New session
# ─────────────────────────────────────────────
else:
    st.header("New Research Session")
    st.caption("Round 1 → Round 2 critique → Round 3 final positions → Synthesis")

    prompt = st.text_area(
        "Your research prompt",
        height=130,
        placeholder="e.g. What are the most high-leverage, non-obvious ways a hedge fund analyst should be using LLMs in their daily workflow right now?"
    )

    run_btn = st.button("⚖️ Convene the Council", type="primary", use_container_width=True)

    if run_btn:
        # ── Validation ──────────────────────────
        if not prompt.strip():
            st.error("Please enter a prompt.")
            st.stop()

        missing = []
        if not anthropic_key:
            missing.append("Anthropic (Claude)")
        if not openai_key:
            missing.append("OpenAI (ChatGPT)")
        if not grok_key:
            missing.append("xAI (Grok)")
        if missing:
            st.error(f"Missing API keys in the sidebar: {', '.join(missing)}")
            st.stop()

        api = {"anthropic": anthropic_key, "openai": openai_key, "grok": grok_key}
        mdl = {"claude": claude_model, "gpt": gpt_model, "grok": grok_model}
        session = {"prompt": prompt, "synthesizer": synthesizer}

        # ── Round 1 ─────────────────────────────
        st.subheader("🔵 Round 1 — Independent Answers")
        st.caption("All three models answer your question with no knowledge of each other's response.")
        with st.spinner("Querying all three models in parallel…"):
            r1 = run_parallel({
                "Claude":  lambda: query_claude(prompt, api["anthropic"], mdl["claude"], max_tokens),
                "ChatGPT": lambda: query_gpt(prompt, api["openai"], mdl["gpt"], max_tokens),
                "Grok":    lambda: query_grok(prompt, api["grok"], mdl["grok"], max_tokens),
            })
        session["round1"] = r1
        show_three_cols(r1)

        # ── Round 2 ─────────────────────────────
        st.divider()
        st.subheader("🟠 Round 2 — Cross-Model Critique")
        st.caption("Each model reads the other two's Round 1 answers and refines its position.")

        p_c2 = build_critique_prompt(prompt, r1["Claude"],  {"ChatGPT": r1["ChatGPT"], "Grok": r1["Grok"]})
        p_g2 = build_critique_prompt(prompt, r1["ChatGPT"], {"Claude": r1["Claude"],   "Grok": r1["Grok"]})
        p_k2 = build_critique_prompt(prompt, r1["Grok"],    {"Claude": r1["Claude"],   "ChatGPT": r1["ChatGPT"]})

        with st.spinner("Each model reviewing the others' answers…"):
            r2 = run_parallel({
                "Claude":  lambda: query_claude(p_c2, api["anthropic"], mdl["claude"], max_tokens),
                "ChatGPT": lambda: query_gpt(p_g2,   api["openai"],     mdl["gpt"],   max_tokens),
                "Grok":    lambda: query_grok(p_k2,   api["grok"],       mdl["grok"],  max_tokens),
            })
        session["round2"] = r2
        show_three_cols(r2)

        # ── Round 3 ─────────────────────────────
        st.divider()
        st.subheader("🔁 Round 3 — Final Positions")
        st.caption("Each model has now seen the full debate and gives its definitive final answer.")

        p_c3 = build_final_position_prompt(
            prompt,
            r1["Claude"],  {"ChatGPT": r1["ChatGPT"], "Grok": r1["Grok"]},
            r2["Claude"],  {"ChatGPT": r2["ChatGPT"], "Grok": r2["Grok"]}
        )
        p_g3 = build_final_position_prompt(
            prompt,
            r1["ChatGPT"], {"Claude": r1["Claude"], "Grok": r1["Grok"]},
            r2["ChatGPT"], {"Claude": r2["Claude"], "Grok": r2["Grok"]}
        )
        p_k3 = build_final_position_prompt(
            prompt,
            r1["Grok"],    {"Claude": r1["Claude"], "ChatGPT": r1["ChatGPT"]},
            r2["Grok"],    {"Claude": r2["Claude"], "ChatGPT": r2["ChatGPT"]}
        )

        with st.spinner("Models giving their definitive final positions…"):
            r3 = run_parallel({
                "Claude":  lambda: query_claude(p_c3, api["anthropic"], mdl["claude"], max_tokens),
                "ChatGPT": lambda: query_gpt(p_g3,   api["openai"],     mdl["gpt"],   max_tokens),
                "Grok":    lambda: query_grok(p_k3,   api["grok"],       mdl["grok"],  max_tokens),
            })
        session["round3"] = r3
        show_three_cols(r3)

        # ── Synthesis ────────────────────────────
        st.divider()
        st.subheader(f"⚡ Final Synthesis — by {synthesizer}")
        st.caption("One model reads the entire debate and writes the research report.")

        synth_p = build_synthesis_prompt(prompt, r1, r2, r3, synthesizer)
        with st.spinner(f"Generating synthesis with {synthesizer}…"):
            if synthesizer == "Claude":
                synthesis = query_claude(synth_p, api["anthropic"], mdl["claude"], max_tokens)
            elif synthesizer == "ChatGPT":
                synthesis = query_gpt(synth_p, api["openai"], mdl["gpt"], max_tokens)
            else:
                synthesis = query_grok(synth_p, api["grok"], mdl["grok"], max_tokens)

        session["synthesis"] = synthesis
        with st.container(border=True):
            st.markdown(synthesis)

        # ── Save & wrap up ───────────────────────
        ts = save_session(session)
        st.success(f"✅ Session saved to history!")

        st.download_button(
            "📥 Download Full Session",
            build_export(session),
            file_name=f"council_{ts}.md",
            mime="text/markdown",
            use_container_width=True
        )
