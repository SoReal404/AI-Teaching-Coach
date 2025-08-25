"""
Neon Tutor – Local UI for a Gemini-powered study agent

✨ Upgrades in this version:
1) Multiple files upload (PDF/TXT/MD)
2) Persistent settings in config.json (API key, model, temp, tokens, thinking budget, last subject)
3) Memorize page with flashcards + spaced repetition (I know / I forgot) + auto-mnemonics
4) TTS speaker (offline, pyttsx3) to read subject or AI output with natural pacing
5) Notes & history kept locally with delete controls
6) 'Test Only' prompt constrained (questions + answer key only)
"""

from __future__ import annotations
import os
import re
import io
import json
from typing import Dict, List, Optional, Tuple
import textwrap
import datetime
from datetime import datetime as dt, timedelta
import concurrent.futures
import traceback

import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types as genai_types

# ---------- UI / THEME ----------
NEON_CSS = """
<style>
:root {
  --bg: #0c1020;           /* deep space */
  --panel: #11162a;        /* card panels */
  --text: #e6eaf2;         /* high-contrast text */
  --muted: #a8b0c7;
  --neon: #7c4dff;         /* violet neon */
  --neon2: #00e5ff;        /* cyan neon */
}
section.main > div { background: var(--bg); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b0f1f 0%, #0c1020 100%); }
.block-container { padding-top: 1.2rem; }
h1, h2, h3, h4 { color: var(--text) !important; text-shadow: 0 0 12px rgba(124,77,255,0.25); }
body, p, span, label, div { color: var(--text); }
.stTextInput > div > div > input,
.stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div,
.stFileUploader { background: var(--panel) !important; color: var(--text) !important; border-radius: 14px; border: 1px solid #1c2342; }
.stButton button {
  background: radial-gradient(120% 120% at 50% 0%, rgba(124,77,255,0.25), rgba(0,229,255,0.15) 60%, transparent), var(--panel);
  color: #fff; border: 1px solid #2a2f55; border-radius: 14px; padding: 0.55rem 1rem; font-weight: 600;
  box-shadow: 0 0 10px rgba(124,77,255,0.25), inset 0 0 30px rgba(0,229,255,0.06);
}
.stButton button:hover { filter: brightness(1.1); box-shadow: 0 0 16px rgba(0,229,255,0.35); }
.stTabs [data-baseweb="tab-list"] { gap: .5rem; }
.stTabs [data-baseweb="tab"] { background: var(--panel); border: 1px solid #232a4a; border-radius: 12px; }
pre, code { background: #0a0f1e !important; color: #d6dcff !important; border-radius: 12px; }
hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, var(--neon2), transparent); }
.badge { display:inline-block; border:1px solid #2a2f55; background: #121735; color:#cfe8ff; padding:.25rem .5rem; border-radius:999px; font-size:.75rem; }
.small { color: var(--muted); font-size: .85rem; }
.card {
  background: var(--panel); border:1px solid #232a4a; border-radius:14px; padding: .75rem 1rem;
  box-shadow: 0 0 10px rgba(124,77,255,0.12), inset 0 0 10px rgba(0,229,255,0.04);
}
</style>
"""

# ---------- FILES / CONFIG ----------
DATA_FILE = "neon_tutor_data.json"     # subjects, notes, history, flashcards
CONFIG_FILE = "config.json"            # api key + model + sliders

st.set_page_config(page_title="K(demo) The Tutor – AI Study Agent", page_icon="📘", layout="wide")
st.markdown(NEON_CSS, unsafe_allow_html=True)

# ---------- PERSISTENCE ----------
def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def init_session():
    if "subjects" not in st.session_state:
        st.session_state.subjects = load_json(DATA_FILE, {})
    if "config" not in st.session_state:
        st.session_state.config = load_json(CONFIG_FILE, {
            "api_key": os.getenv("GEMINI_API_KEY", ""),
            "model_name": "gemini-1.5-pro",
            "max_output_tokens": 1024,
            "temperature": 0.6,
            "thinking_budget": 0,
            "last_subject": "math",
            "tts_rate": 175,
            "tts_volume": 0.9,
            "voice_id": None
        })
    if "current_subject" not in st.session_state:
        st.session_state.current_subject = st.session_state.config.get("last_subject", "math")

init_session()

def save_all():
    save_json(DATA_FILE, st.session_state.subjects)
    st.session_state.config["last_subject"] = st.session_state.current_subject
    save_json(CONFIG_FILE, st.session_state.config)

# ---------- GEMINI ----------
def ensure_client(api_key: str) -> Optional[genai.Client]:
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Gemini client error: {e}")
        return None

SYSTEM_PROMPT = """
You are *K Tutor*, an elite study coach. Goals for EVERY prompt:
1) Explain simply first, then deepen (use headings).
2) Provide memory tricks: mnemonics, chunking, analogies, visuals.
3) Use modern methods: Active recall, Spaced repetition, Interleaving, Dual coding, Feynman technique, Deliberate practice, Retrieval practice.
4) Create a practice test (5 questions: MCQ, fill-in, derivation) + an Answer Key.
5) Use Markdown + LaTeX ($$...$$). You may include Graphviz blocks for simple diagrams.
6) Be concise but precise.
"""

def build_messages(resource_text: str, user_goal: str, subject: str) -> List:
    user_msg = f"""
[Subject] {subject}
[Goal] {user_goal}
[Resource]
{resource_text}
"""
    return [SYSTEM_PROMPT, user_msg]

def call_gemini(client: genai.Client, model: str, contents: List[str], *, temp: float, max_tokens: int, thinking_budget: int) -> str:
    cfg = genai_types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=temp)
    if "2.5" in model and thinking_budget:
        cfg = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temp,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=thinking_budget)
        )
    resp = client.models.generate_content(model=model, contents=contents, config=cfg)
    return getattr(resp, "text", str(resp))

def safe_call_gemini(client, model, contents, *, temp, max_tokens, thinking_budget):
    """Call Gemini and return text or log & return empty string on failure."""
    try:
        resp_text = call_gemini(client, model, contents, temp=temp, max_tokens=max_tokens, thinking_budget=thinking_budget)
        if resp_text is None:
            return ""
        return str(resp_text)
    except Exception as e:
        # log debug info
        try:
            with open("gemini_errors.log", "a", encoding="utf-8") as lf:
                lf.write(f"\n\n--- {dt.now().isoformat()} ---\nException:\n")
                lf.write(traceback.format_exc())
        except Exception:
            pass
        st.error(f"Model call failed: {e}")
        return ""

# ---------- HELPERS ----------
def load_pdf_text(file: io.BytesIO, max_pages: int = 30) -> str:
    try:
        reader = PdfReader(file)
        pages = min(len(reader.pages), max_pages)
        return "\n\n".join([reader.pages[i].extract_text() or "" for i in range(pages)])
    except Exception as e:
        return f"[PDF read error: {e}]"

GRAPHVIZ_RE = re.compile(r"```graphviz\n(.*?)```", re.DOTALL)

def render_markdown_with_extras(md: str):
    last_end = 0
    for m in GRAPHVIZ_RE.finditer(md):
        before = md[last_end:m.start()]
        if before.strip():
            st.markdown(before)
        code = m.group(1)
        try:
            st.graphviz_chart(code)
        except Exception as e:
            st.warning(f"Graphviz parse error: {e}")
            st.code(code, language="dot")
        last_end = m.end()
    tail = md[last_end:]
    if tail.strip():
        st.markdown(tail)

def subj_obj(subject: str) -> Dict:
    st.session_state.subjects.setdefault(subject, {"notes": [], "history": [], "flashcards": []})
    return st.session_state.subjects[subject]

def add_note(subject: str, content: str):
    p = subj_obj(subject)
    p["notes"].append({"timestamp": dt.now().isoformat(), "content": content})
    save_all()

def delete_note(subject: str, idx_from_end: int):
    p = subj_obj(subject)
    real_idx = len(p["notes"]) - idx_from_end - 1
    if 0 <= real_idx < len(p["notes"]):
        p["notes"].pop(real_idx)
        save_all()

# ---------- FLASHCARDS (SM2-lite) ----------
# Each card: {"front": str, "back": str, "ef": 2.5, "interval": 0, "due": "ISO", "last_rating": int}
def add_flashcard(subject: str, front: str, back: str):
    p = subj_obj(subject)
    p["flashcards"].append({
        "front": front.strip(),
        "back": back.strip(),
        "ef": 2.5,
        "interval": 0,
        "due": dt.now().isoformat(),
        "last_rating": None
    })
    save_all()

def rate_flashcard(card: Dict, rating: int):
    # rating: 0=again (forgot), 3=hard, 4=good, 5=easy
    ef = max(1.3, card.get("ef", 2.5) + (0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02)))
    interval = card.get("interval", 0)
    if rating < 3:
        interval = 1
    else:
        if interval == 0:
            interval = 1
        elif interval == 1:
            interval = 6
        else:
            interval = int(round(interval * ef))
    card["ef"] = ef
    card["interval"] = interval
    card["due"] = (dt.now() + timedelta(days=interval)).isoformat()
    card["last_rating"] = rating

def next_due_cards(cards: List[Dict], limit: int = 10) -> List[Dict]:
    now = dt.now()
    due = [c for c in cards if dt.fromisoformat(c["due"]) <= now]
    return due[:limit] if due else sorted(cards, key=lambda c: c["due"])[:limit]

# ---------- TTS (pyttsx3) ----------
# We generate WAV files locally and stream them with st.audio.
def tts_speak(text: str, rate: int, volume: float, voice_id: Optional[str]) -> Optional[str]:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        if rate: engine.setProperty('rate', rate)
        if volume: engine.setProperty('volume', volume)
        if voice_id:
            engine.setProperty('voice', voice_id)
        filename = f"tts_{int(dt.now().timestamp())}.wav"
        engine.save_to_file(text, filename)
        engine.runAndWait()
        return filename
    except Exception as e:
        st.warning(f"TTS not available: {e}")
        return None

def list_voices() -> List[Tuple[str, str]]:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        out = []
        for v in engine.getProperty('voices'):
            out.append((v.id, getattr(v, "name", v.id)))
        return out
    except Exception:
        return []

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Settings")

# Load persisted config defaults
cfg = st.session_state.config
api_key = st.sidebar.text_input("Gemini API Key (saved locally)", value=cfg.get("api_key", ""), type="password")
model_name = st.sidebar.selectbox(
    "Model",
    options=["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    index=["gemini-1.5-pro","gemini-1.5-flash","gemini-2.5-flash","gemini-2.5-pro"].index(cfg.get("model_name","gemini-1.5-pro"))
)
max_output_tokens = st.sidebar.slider("Max output tokens", 256, 4096, cfg.get("max_output_tokens", 1024), step=128)
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, cfg.get("temperature", 0.6), step=0.05)
thinking_budget = st.sidebar.slider("Thinking budget (2.5 only)", 0, 8000, cfg.get("thinking_budget", 0), step=500)

st.sidebar.markdown("---")
st.sidebar.subheader("🎧 TTS")
voice_options = list_voices()
voice_names = [name for _, name in voice_options] if voice_options else []
voice_id = cfg.get("voice_id", voice_options[0][0] if voice_options else None)
selected_voice = st.sidebar.selectbox("Voice", voice_names, index=0 if voice_names else 0) if voice_names else None
if voice_names:
    voice_id = voice_options[voice_names.index(selected_voice)][0]
tts_rate = st.sidebar.slider("Rate", 120, 220, cfg.get("tts_rate", 175), step=5)
tts_volume = st.sidebar.slider("Volume", 0.2, 1.0, cfg.get("tts_volume", 0.9), step=0.05)

# persist config immediately when changed
st.session_state.config.update({
    "api_key": api_key,
    "model_name": model_name,
    "max_output_tokens": max_output_tokens,
    "temperature": temperature,
    "thinking_budget": thinking_budget,
    "voice_id": voice_id,
    "tts_rate": tts_rate,
    "tts_volume": tts_volume
})
save_json(CONFIG_FILE, st.session_state.config)

st.sidebar.markdown("---")
subject = st.sidebar.text_input("Current subject", value=st.session_state.get("current_subject", "math"))
if st.sidebar.button("➕ Add / Switch subject"):
    st.session_state.current_subject = subject
    subj_obj(subject)
    save_all()

# ---------- HEADER ----------
col_a, col_b = st.columns([1.1, 1])
with col_a:
    st.title("K(demo) The Tutor")
    st.caption("Explain • Memorize • Test — powered by Gemini")
with col_b:
    st.markdown("<div class='badge'>Dark • Neon • Local</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- INPUTS (MULTI-FILE + GOAL) ----------
left, right = st.columns([1, 1])
with left:
    subject = st.text_input("Subject", value=st.session_state.get("current_subject", "math"))
    user_goal = st.text_area("What do you want to learn?", placeholder="e.g., Understand eigenvalues and how to compute them.", height=100)
with right:
    uploads = st.file_uploader("Upload resources (PDF/TXT/MD) — multiple allowed", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True)
    pasted_text = st.text_area("…or paste text / notes", height=160)

# Combine all resources
resource_texts = []
if uploads:
    for up in uploads:
        ext = up.name.split(".")[-1].lower()
        if ext == "pdf":
            resource_texts.append(load_pdf_text(up))
        else:
            resource_texts.append(up.read().decode("utf-8", errors="ignore"))
if pasted_text:
    resource_texts.append(pasted_text)
resource_text = "\n\n---\n\n".join([t for t in resource_texts if t])

# ---------- ACTIONS ----------
col1, col2, col3, col4 = st.columns([1,1,1,1])
ask = col1.button("✨ Explain & Teach")
quiz_only = col2.button("🧪 Make a Test Only")
save_res_note = col3.button("📌 Save Resource Text")
gen_cards = col4.button("🧩 Generate Flashcards")

if save_res_note and resource_text:
    add_note(subject, resource_text[:4000])
    st.success("Saved resource text to subject notes (local JSON).")

# ---------- TABS ----------
learn_tab, test_tab, memorize_tab, history_tab = st.tabs(["📘 Learn", "🧪 Test", "🧠 Memorize", "🗂 Notes & History"])

# ----- LEARN -----
with learn_tab:
    speak_col, _ = st.columns([1,3])
    if ask:
        if not api_key:
            st.error("Add your Gemini API key in the sidebar.")
        elif not (resource_text or user_goal):
            st.warning("Please upload/paste a resource or describe your goal.")
        else:
            client = ensure_client(api_key)
            if client:
                msgs = build_messages(resource_text, user_goal or "Explain simply", subject)
                with st.spinner("Thinking…"):
                    out = safe_call_gemini(
                        client, model_name, msgs,
                        temp=temperature, max_tokens=max_output_tokens, thinking_budget=thinking_budget
                    )

                add_note(subject, out)
                subj_obj(subject)["history"].append(("learn", out))
                save_all()
                render_markdown_with_extras(out)
                if speak_col.button("🔊 Read this"):
                    wav = tts_speak(f"Subject: {subject}. {out}", tts_rate, tts_volume, voice_id)
                    if wav: st.audio(wav)

# ----- TEST -----
with test_tab:
    speak_col2, _ = st.columns([1,3])
    if quiz_only:
        if not api_key:
            st.error("Add your Gemini API key in the sidebar.")
        else:
            client = ensure_client(api_key)
            if client:
                quiz_prompt = textwrap.dedent(f"""
                ONLY generate a **Practice Test** with 5 questions (mix MCQ, fill-in, derivation) for subject: {subject}.
                Include a collapsible **Answer Key** at the end.
                NO explanations, NO extra commentary.
                """)
                # Run the model call with a timeout so the UI doesn't hang forever
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                model_input = [quiz_prompt, resource_text or user_goal or "Focus on fundamentals."]
                future = executor.submit(safe_call_gemini, client, model_name, model_input,
                                         temp=temperature, max_tokens=max_output_tokens, thinking_budget=thinking_budget)
                try:
                    out = future.result(timeout=35)  # seconds
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    st.error("Model request timed out ( > 35s ). Try reducing Max tokens or use a simpler prompt.")
                    out = None
                except Exception as e:
                    st.error(f"Error calling model: {e}")
                    out = None
                finally:
                    executor.shutdown(wait=False)

                # If call failed, try a short fallback prompt synchronously (less tokens)
                if not out:
                    try:
                        fallback_prompt = textwrap.dedent(f"""
                        Generate 5 short practice questions (questions only, no answers) for: {subject}.
                        Keep them concise.
                        """)
                        with st.spinner("Trying fallback test prompt..."):
                            out = safe_call_gemini(client, model_name, [fallback_prompt], temp=0.2, max_tokens=512, thinking_budget=0)
                    except Exception as e:
                        st.error(f"Fallback also failed: {e}")
                        out = None

                if out:
                    add_note(subject, out)
                    subj_obj(subject)["history"].append(("test", out))
                    save_all()
                    render_markdown_with_extras(out)
                    if speak_col2.button("🔊 Read test"):
                        wav = tts_speak(out, tts_rate, tts_volume, voice_id)
                        if wav: st.audio(wav)
                else:
                    st.info("Could not generate the test. Try again with a different prompt or check your API quota/connection.")

# ----- MEMORIZE -----
with memorize_tab:
    st.subheader("🧠 Flashcards (Spaced Repetition)")

    # Add manually
    with st.expander("➕ Create a card"):
        c1, c2 = st.columns(2)
        front = c1.text_area("Front (prompt)", height=100, placeholder="e.g., What is the derivative of x^n?")
        back = c2.text_area("Back (answer / trick)", height=100, placeholder="e.g., n·x^(n-1). Trick: bring power down, minus one.")
        if st.button("Add Card"):
            if front.strip() and back.strip():
                add_flashcard(subject, front, back)
                st.success("Card added.")
            else:
                st.warning("Front and back required.")

    # Auto-generate from resources
    if gen_cards:
        if not api_key:
            st.error("Add your Gemini API key in the sidebar.")
        elif not resource_text:
            st.warning("Upload files or paste text to extract flashcards from.")
        else:
            client = ensure_client(api_key)
            if client:
                prompt = textwrap.dedent(f"""
                From the following study material, generate 8 concise flashcards as JSON list.
                Each item: {{ "front": "...", "back": "... (include a short mnemonic if helpful)" }}
                Keep them short and clear.

                MATERIAL:
                {resource_text[:6000]}
                """)
                # Create flashcards robustly (guard the model call + tolerant JSON parsing)
                import json as _json
                import re as _re
                cards_added = 0
                parsed = None
                out = ""

                try:
                    with st.spinner("Creating flashcards…"):
                        out = safe_call_gemini(client, model_name, [prompt], temp=0.5, max_tokens=1024, thinking_budget=0) or ""
                except Exception as e:
                    st.error(f"Error while calling model for flashcards: {e}")
                    st.warning("Check API key, quota, or network. See flashcards_debug.log for details.")
                    # write raw exception + partial output to debug file
                    try:
                        with open("flashcards_debug.log", "a", encoding="utf-8") as lf:
                            lf.write(f"\n\n--- {dt.now().isoformat()} ---\nException:\n")
                            lf.write(traceback.format_exc())
                            lf.write("\nRaw output (if any):\n")
                            lf.write(str(out)[:8000])
                    except Exception:
                        pass
                    out = ""

                # Only try parsing if we have text
                if out:
                    # 1) direct parse attempt
                    try:
                        parsed = _json.loads(out)
                    except Exception:
                        parsed = None

                    # 2) try to extract a JSON array substring like [ ... ] using regex
                    if parsed is None:
                        m = _re.search(r"(\[\s*\{.*\}\s*\])", out, _re.DOTALL)
                        if m:
                            try:
                                parsed = _json.loads(m.group(1))
                            except Exception:
                                parsed = None

                    # 3) fallback: find multiple JSON objects and try to decode them individually
                    if parsed is None:
                        objs = _re.findall(r"(\{(?:[^{}]|\{[^{}]*\})*\})", out, _re.DOTALL)
                        if objs:
                            tmp = []
                            for o in objs:
                                try:
                                    tmp.append(_json.loads(o))
                                except Exception:
                                    # skip poorly-formed object
                                    pass
                            if tmp:
                                parsed = tmp

                # If we have parsed data, add cards
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "front" in item and "back" in item:
                            add_flashcard(subject, item["front"], item["back"])
                            cards_added += 1

                if cards_added:
                    st.success(f"Added {cards_added} flashcards.")
                else:
                    if out:
                        st.info("Couldn't parse JSON from model. Here is the model output (truncated) to help debugging:")
                        st.code(out[:3000] + ("... (truncated)" if len(out) > 3000 else ""))
                    else:
                        st.info("No model output produced for flashcard generation.")
                    st.warning("Try simplifying the resource text, reduce content length, or add cards manually.")

    # Review queue
    cards = subj_obj(subject)["flashcards"]
    if not cards:
        st.info("No flashcards yet. Add or generate some above.")
    else:
        due_cards = next_due_cards(cards, limit=10)
        if not due_cards:
            st.success("All caught up! No cards due. 🎉")
        else:
            for i, card in enumerate(due_cards):
                with st.container():
                    st.markdown(f"**Card {i+1}/{len(due_cards)}** — due: {card['due']}")
                    with st.expander("👁️ Show Answer"):
                        st.write(card["back"])
                        # TTS for card
                        tcol1, tcol2, _ = st.columns([1,1,6])
                        if tcol1.button("🔊 Read Front", key=f"tts_front_{i}"):
                            wav = tts_speak(card["front"], tts_rate, tts_volume, voice_id)
                            if wav: st.audio(wav)
                        if tcol2.button("🔊 Read Back", key=f"tts_back_{i}"):
                            wav = tts_speak(card["back"], tts_rate, tts_volume, voice_id)
                            if wav: st.audio(wav)
                    c1, c2, c3, c4 = st.columns(4)
                    if c1.button("I forgot", key=f"rate0_{i}"):
                        rate_flashcard(card, 0)
                        # Optional: Ask model for a mnemonic on fail
                        if api_key:
                            client = ensure_client(api_key)
                            if client:
                                mnemo_prompt = f"Give a short memorable trick to remember this: FRONT: {card['front']} BACK: {card['back']}"
                                try:
                                    trick = safe_call_gemini(client, model_name, [mnemo_prompt], temp=0.7, max_tokens=200, thinking_budget=0)
                                    st.info(f"Mnemonic idea: {trick}")
                                except Exception:
                                    pass
                        save_all()
                        st.rerun()
                    if c2.button("Hard", key=f"rate3_{i}"):
                        rate_flashcard(card, 3); save_all(); st.rerun()
                    if c3.button("Good", key=f"rate4_{i}"):
                        rate_flashcard(card, 4); save_all(); st.rerun()
                    if c4.button("Easy", key=f"rate5_{i}"):
                        rate_flashcard(card, 5); save_all(); st.rerun()
                    st.markdown("<hr/>", unsafe_allow_html=True)

# ----- NOTES & HISTORY -----
with history_tab:
    prof = subj_obj(subject)
    st.subheader("📒 Notes")
    if not prof.get("notes"):
        st.info("No notes yet for this subject.")
    else:
        for idx, note in enumerate(reversed(prof["notes"])):
            with st.expander(f"📝 Note {len(prof['notes'])-idx} — {note['timestamp']}"):
                st.write(note["content"])
                ncol1, ncol2 = st.columns([1,6])
                if ncol1.button("🗑 Delete", key=f"del_note_{idx}"):
                    delete_note(subject, idx)
                    st.rerun()
                if ncol2.button("🔊 Read", key=f"tts_note_{idx}"):
                    wav = tts_speak(note["content"], tts_rate, tts_volume, voice_id)
                    if wav: st.audio(wav)

    st.subheader("🗂 History")
    if not prof.get("history"):
        st.info("No history yet.")
    else:
        for idx, (kind, content) in enumerate(reversed(prof["history"])):
            with st.expander(f"{kind.upper()} #{len(prof['history'])-idx}"):
                render_markdown_with_extras(content)
                if st.button("🔊 Read", key=f"tts_hist_{idx}"):
                    wav = tts_speak(content, tts_rate, tts_volume, voice_id)
                    if wav: st.audio(wav)

st.markdown("---")
st.caption("Tip: Include formulas; Neon Tutor renders LaTeX and simple Graphviz diagrams.")
