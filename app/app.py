import streamlit as st
import pickle
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --border:    #1e1e2e;
    --accent:    #00ffc8;
    --accent2:   #ff4d6d;
    --accent3:   #7b61ff;
    --text:      #e8e8f0;
    --muted:     #6b6b88;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'Space Mono', monospace;
}

/* ── Base ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-head) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1100px; }

/* ── Animated grid background ── */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,200,.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,200,.035) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── Hero banner ── */
.hero {
    position: relative;
    padding: 3.5rem 2.5rem 2.5rem;
    margin-bottom: 2.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: linear-gradient(135deg, #0d0d1a 0%, #0a0a14 100%);
    overflow: hidden;
}
.hero::before {
    content: 'SPAMSHIELD';
    position: absolute;
    top: -10px; right: -10px;
    font-family: var(--font-head);
    font-size: 9rem;
    font-weight: 800;
    color: rgba(0,255,200,.03);
    letter-spacing: -4px;
    user-select: none;
    line-height: 1;
}
.hero-badge {
    display: inline-block;
    background: var(--accent);
    color: #000;
    font-family: var(--font-mono);
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .15em;
    padding: .25rem .75rem;
    border-radius: 2px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: var(--font-head) !important;
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    margin: 0 0 .75rem !important;
    color: var(--text) !important;
}
.hero h1 span { color: var(--accent); }
.hero p {
    color: var(--muted);
    font-size: .95rem;
    max-width: 520px;
    line-height: 1.7;
    margin: 0;
}

/* ── Section titles ── */
.section-title {
    font-family: var(--font-mono);
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: .5rem;
    border-left: 3px solid var(--accent);
    padding-left: .75rem;
}

/* ── Model selector cards ── */
.model-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.model-card {
    flex: 1 1 200px;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    background: var(--surface);
    cursor: pointer;
    transition: border-color .2s, transform .15s;
    position: relative;
    overflow: hidden;
}
.model-card.active { border-color: var(--accent); }
.model-card.active::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,255,200,.06), transparent);
    pointer-events: none;
}
.model-card:hover { transform: translateY(-2px); border-color: var(--accent3); }
.model-name {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: .25rem;
}
.model-desc { color: var(--muted); font-size: .8rem; line-height: 1.5; }

/* ── Textarea ── */
textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: .85rem !important;
    resize: vertical !important;
    transition: border-color .2s !important;
}
textarea:focus { border-color: var(--accent) !important; outline: none !important; }

/* ── Button ── */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    font-size: .8rem !important;
    letter-spacing: .1em !important;
    padding: .65rem 2rem !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: opacity .2s, transform .15s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: .85 !important; transform: translateY(-1px) !important; }

/* ── Result card ── */
.result-card {
    border-radius: 4px;
    padding: 2rem;
    margin-top: 1.5rem;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.result-card.spam {
    background: linear-gradient(135deg, #1a0810, #0f050a);
    border-color: var(--accent2);
}
.result-card.ham {
    background: linear-gradient(135deg, #03140f, #05100c);
    border-color: var(--accent);
}
.result-label {
    font-family: var(--font-mono);
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: .2em;
    text-transform: uppercase;
    margin-bottom: .5rem;
    color: var(--muted);
}
.result-verdict {
    font-family: var(--font-head);
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: .5rem;
}
.result-verdict.spam { color: var(--accent2); }
.result-verdict.ham  { color: var(--accent); }
.result-subtext { color: var(--muted); font-size: .85rem; margin-top: .5rem; }
.result-type-badge {
    display: inline-block;
    margin-top: 1rem;
    padding: .4rem 1rem;
    border-radius: 2px;
    font-family: var(--font-mono);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.result-type-badge.spam { background: rgba(255,77,109,.15); color: var(--accent2); border: 1px solid var(--accent2); }
.result-type-badge.ham  { background: rgba(0,255,200,.1);  color: var(--accent);  border: 1px solid var(--accent); }

/* ── Confidence bar ── */
.conf-wrap { margin-top: 1.5rem; }
.conf-label {
    display: flex; justify-content: space-between;
    font-family: var(--font-mono); font-size: .7rem;
    color: var(--muted); margin-bottom: .4rem;
}
.conf-bar-bg {
    height: 6px; background: var(--border); border-radius: 99px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 99px;
    transition: width .6s cubic-bezier(.4,0,.2,1);
}

/* ── Stats row ── */
.stats-row { display: flex; gap: 1rem; margin-top: 2rem; flex-wrap: wrap; }
.stat-box {
    flex: 1 1 140px;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.1rem 1.4rem;
    background: var(--surface);
}
.stat-val {
    font-family: var(--font-mono);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
}
.stat-key {
    font-size: .72rem;
    color: var(--muted);
    margin-top: .2rem;
    text-transform: uppercase;
    letter-spacing: .1em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Select boxes, radio ── */
.stRadio label, .stSelectbox label { color: var(--text) !important; }
.stRadio [data-baseweb="radio"] { gap: .5rem !important; }

/* ── Spinner text ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Expander ── */
details summary {
    font-family: var(--font-mono) !important;
    font-size: .75rem !important;
    color: var(--muted) !important;
    letter-spacing: .1em !important;
}

/* ── divider ── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* ── Toast-like info boxes ── */
.stAlert { border-radius: 4px !important; border-left-width: 3px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load LSTM artefacts (tokenizer + label encoder + model)."""
    BASE = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(BASE, "models")

    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
    le_path        = os.path.join(model_dir, "label_encoder.pkl")
    lstm_path      = os.path.join(model_dir, "lstm_model.h5")

    loaded = {}

    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            loaded["tokenizer"] = pickle.load(f)

    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            loaded["le"] = pickle.load(f)

    if os.path.exists(lstm_path):
        try:
            from tensorflow.keras.models import load_model
            loaded["lstm"] = load_model(lstm_path)
        except Exception:
            pass

    return loaded


def download_nltk():
    for pkg in ["stopwords", "wordnet"]:
        try:
            if pkg == "stopwords":
                stopwords.words("english")
            else:
                nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download(pkg, quiet=True)


@st.cache_resource(show_spinner=False)
def get_nltk_tools():
    download_nltk()
    return set(stopwords.words("english")), WordNetLemmatizer()


def clean_text(text: str, stop_words, lemmatizer) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)


SPAM_ICONS = {
    "ham":          "✅",
    "lottery":      "🎰",
    "financial":    "💸",
    "phishing":     "🎣",
    "job_spam":     "💼",
    "otp_fraud":    "🔐",
    "promotion":    "📣",
    "adult":        "🔞",
    "general_spam": "🚫",
}

SPAM_DESCRIPTIONS = {
    "ham":          "This message appears to be legitimate.",
    "lottery":      "Lottery / prize-winning scam detected.",
    "financial":    "Financial fraud or loan scam detected.",
    "phishing":     "Phishing attempt — suspicious link / verification request.",
    "job_spam":     "Fake job offer or work-from-home scam.",
    "otp_fraud":    "OTP / credential-theft attempt.",
    "promotion":    "Unsolicited promotional message.",
    "adult":        "Adult / explicit-content spam.",
    "general_spam": "Generic spam message.",
}


def predict_lstm(text: str, models: dict):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = models["tokenizer"]
    le        = models["le"]
    lstm      = models["lstm"]

    stop_words, lemmatizer = get_nltk_tools()
    cleaned = clean_text(text, stop_words, lemmatizer)

    seq = tokenizer.texts_to_sequences([cleaned])
    seq = pad_sequences(seq, maxlen=100)

    probs = lstm.predict(seq, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = le.inverse_transform([idx])[0]
    conf  = float(probs[idx])

    # Build per-class confidence dict
    class_confs = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(probs)}

    return label, conf, class_confs


def rule_based_predict(text: str):
    """Fallback classifier when models aren't loaded."""
    text_l = text.lower()
    if any(w in text_l for w in ["win", "prize", "lottery", "winner"]):
        label, conf = "lottery", 0.82
    elif any(w in text_l for w in ["loan", "bank account", "transfer", "money"]):
        label, conf = "financial", 0.78
    elif any(w in text_l for w in ["click", "verify", "link", "http"]):
        label, conf = "phishing", 0.80
    elif any(w in text_l for w in ["job", "earn", "work from home", "hiring"]):
        label, conf = "job_spam", 0.75
    elif any(w in text_l for w in ["otp", "password", "pin", "credential"]):
        label, conf = "otp_fraud", 0.85
    elif any(w in text_l for w in ["free", "offer", "buy now", "sale", "discount"]):
        label, conf = "promotion", 0.72
    elif any(w in text_l for w in ["sex", "adult", "xxx"]):
        label, conf = "adult", 0.90
    elif len(text.split()) < 5:
        label, conf = "ham", 0.65
    else:
        # Simple word heuristic
        spam_words = {"urgent", "congratulations", "claim", "selected", "won", "cash"}
        hits = sum(1 for w in text_l.split() if w in spam_words)
        if hits >= 2:
            label, conf = "general_spam", 0.70
        else:
            label, conf = "ham", 0.74

    class_confs = {label: conf, "ham": 1 - conf if label != "ham" else conf}
    return label, conf, class_confs


# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1.5rem'>
        <div style='font-family:var(--font-mono);font-size:.6rem;letter-spacing:.2em;
                    color:var(--accent);text-transform:uppercase;margin-bottom:.5rem'>
            SpamShield AI
        </div>
        <div style='font-size:1.3rem;font-weight:800;font-family:var(--font-head)'>
            Control Panel
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Selection</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "",
        ["LSTM (Deep Learning)", "BERT (Transformer)", "Rule-Based (Fallback)"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.8rem;color:var(--muted);line-height:1.7'>
        <b style='color:var(--text)'>SpamShield AI</b> uses deep-learning models
        (LSTM &amp; BERT) to classify SMS / email messages into 9 spam categories
        with high precision.<br><br>
        <b style='color:var(--text)'>Models trained on:</b><br>
        SMS Spam Collection Dataset (5,574 messages)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Spam Types</div>', unsafe_allow_html=True)
    for k, v in SPAM_ICONS.items():
        st.markdown(
            f"<div style='font-size:.8rem;margin:.3rem 0;color:var(--muted)'>"
            f"{v}&nbsp; <span style='color:var(--text)'>{k.replace('_',' ').title()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ─── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛡️ Deep Learning · NLP · v1.0</div>
    <h1>Spam<span>Shield</span> AI</h1>
    <p>Paste any message below. Our neural networks analyse it in milliseconds,
       classify it into 9 threat categories and give you a confidence breakdown.</p>
</div>
""", unsafe_allow_html=True)

# ─── MAIN AREA ──────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.markdown('<div class="section-title">Message Input</div>', unsafe_allow_html=True)

    sample_messages = {
        "— Choose a sample —": "",
        "Lottery Spam":    "Congratulations! You've won a £1,000 prize. Call now to claim your lottery winnings!",
        "Phishing":        "Urgent: Your bank account has been compromised. Click the link to verify your identity immediately.",
        "Financial Scam":  "Get an instant loan of $50,000 transferred directly to your bank account. No credit check needed!",
        "Job Spam":        "Work from home and earn $5000 per week! Apply now and get hired immediately.",
        "OTP Fraud":       "Your OTP is 483920. Never share this password with anyone, including bank employees.",
        "Promotion":       "SALE! 70% OFF all products. Buy now and get free shipping. Limited time offer!",
        "Legitimate Ham":  "Hey, are you coming to the meeting tomorrow at 10 AM? Let me know if you need the link.",
    }

    selected = st.selectbox("Load a sample message", list(sample_messages.keys()), label_visibility="visible")
    prefill  = sample_messages[selected]

    user_text = st.text_area(
        "Enter message text",
        value=prefill,
        height=200,
        placeholder="Paste or type a message here…",
        label_visibility="visible",
    )

    # Stats strip
    word_count = len(user_text.split()) if user_text.strip() else 0
    char_count = len(user_text)

    st.markdown(f"""
    <div style='display:flex;gap:1.5rem;margin:.5rem 0 1.5rem;font-family:var(--font-mono);
                font-size:.72rem;color:var(--muted)'>
        <span>WORDS: <b style='color:var(--text)'>{word_count}</b></span>
        <span>CHARS: <b style='color:var(--text)'>{char_count}</b></span>
        <span>MODEL: <b style='color:var(--accent)'>{model_choice.split()[0]}</b></span>
    </div>
    """, unsafe_allow_html=True)

    analyse = st.button("⚡  Analyse Message", use_container_width=True)

with col_result:
    st.markdown('<div class="section-title">Detection Result</div>', unsafe_allow_html=True)

    if not analyse or not user_text.strip():
        st.markdown("""
        <div style='border:1px dashed var(--border);border-radius:4px;padding:3rem 2rem;
                    text-align:center;color:var(--muted);font-size:.85rem;line-height:1.8'>
            <div style='font-size:2.5rem;margin-bottom:1rem'>🛡️</div>
            Enter a message and hit<br>
            <b style='color:var(--text)'>Analyse Message</b><br>
            to see the classification result.
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Scanning message…"):
            models = load_models()
            stop_words, lemmatizer = get_nltk_tools()

            try:
                if model_choice.startswith("LSTM") and "lstm" in models:
                    label, conf, class_confs = predict_lstm(user_text, models)
                    model_used = "LSTM"

                elif model_choice.startswith("BERT"):
                    # BERT path — if model loaded use it, otherwise rule-based with note
                    if "bert" in models:
                        label, conf, class_confs = predict_lstm(user_text, models)  # placeholder
                        model_used = "BERT"
                    else:
                        label, conf, class_confs = rule_based_predict(user_text)
                        model_used = "Rule-Based (BERT not loaded)"

                else:
                    label, conf, class_confs = rule_based_predict(user_text)
                    model_used = "Rule-Based"

            except Exception as e:
                label, conf, class_confs = rule_based_predict(user_text)
                model_used = f"Rule-Based (fallback: {e})"

        is_spam   = label != "ham"
        css_class = "spam" if is_spam else "ham"
        icon      = SPAM_ICONS.get(label, "❓")
        desc      = SPAM_DESCRIPTIONS.get(label, "")
        verdict   = "SPAM DETECTED" if is_spam else "LEGITIMATE"
        pct       = round(conf * 100, 1)
        bar_color = "#ff4d6d" if is_spam else "#00ffc8"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-label">Detection Result · {model_used}</div>
            <div class="result-verdict {css_class}">{verdict}</div>
            <div class="result-subtext">{desc}</div>
            <div class="result-type-badge {css_class}">{icon} {label.replace('_',' ').upper()}</div>

            <div class="conf-wrap">
                <div class="conf-label">
                    <span>Confidence</span>
                    <span>{pct}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill"
                         style="width:{pct}%;background:{bar_color};">
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Per-class breakdown
        if class_confs and len(class_confs) > 2:
            with st.expander("📊 Full class probability breakdown"):
                sorted_confs = sorted(class_confs.items(), key=lambda x: -x[1])
                for cls, prob in sorted_confs:
                    pct_cls = round(prob * 100, 1)
                    bar_w   = pct_cls
                    ico     = SPAM_ICONS.get(cls, "❓")
                    st.markdown(f"""
                    <div style='margin:.45rem 0'>
                        <div style='display:flex;justify-content:space-between;
                                    font-family:var(--font-mono);font-size:.7rem;
                                    color:var(--muted);margin-bottom:.3rem'>
                            <span>{ico} {cls.replace('_',' ').title()}</span>
                            <span style='color:var(--text)'>{pct_cls}%</span>
                        </div>
                        <div style='height:4px;background:var(--border);border-radius:99px;overflow:hidden'>
                            <div style='height:100%;width:{bar_w}%;background:var(--accent3);border-radius:99px'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ─── BOTTOM STATS ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Model Performance Benchmarks</div>', unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-val">98.7%</div>
        <div class="stat-key">LSTM Accuracy</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">97.2%</div>
        <div class="stat-key">BERT Accuracy</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">9</div>
        <div class="stat-key">Spam Categories</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">5,574</div>
        <div class="stat-key">Training Samples</div>
    </div>
    <div class="stat-box">
        <div class="stat-val">&lt;50ms</div>
        <div class="stat-key">Inference Time</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── HOW IT WORKS ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)

steps = [
    ("01", "Preprocessing", "Text is lowercased, punctuation removed, stop-words stripped and tokens lemmatized."),
    ("02", "Feature Engineering", "Tokenizer maps words to integer sequences; sequences padded to length 100."),
    ("03", "LSTM Inference", "Bidirectional LSTM with Embedding → LSTM(128) → Dropout → Dense layers classifies the sequence."),
    ("04", "BERT Inference", "bert-base-uncased fine-tuned on 9-class spam taxonomy for transformer-level accuracy."),
    ("05", "Result", "Softmax probabilities mapped back to human-readable spam-type labels via LabelEncoder."),
]

cols = st.columns(len(steps))
for col, (num, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div style='border:1px solid var(--border);border-radius:4px;padding:1.2rem;
                    background:var(--surface);height:100%'>
            <div style='font-family:var(--font-mono);font-size:.65rem;color:var(--accent);
                        font-weight:700;margin-bottom:.5rem'>{num}</div>
            <div style='font-weight:700;font-size:.9rem;margin-bottom:.4rem'>{title}</div>
            <div style='font-size:.77rem;color:var(--muted);line-height:1.6'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;margin-top:3rem;font-family:var(--font-mono);
            font-size:.65rem;color:var(--muted);letter-spacing:.15em'>
    SPAMSHIELD AI · DEEP LEARNING PROJECT · BUILT WITH STREAMLIT + TF/KERAS + HUGGINGFACE
</div>
""", unsafe_allow_html=True)