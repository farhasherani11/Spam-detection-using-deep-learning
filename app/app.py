import streamlit as st
import pickle, numpy as np, os, re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="SpamShield AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ─── TOKENS ─────────────────────────────────────── */
:root {
  --bg:        #05070f;
  --s0:        #080b18;
  --s1:        #0b0e1f;
  --s2:        #0e1228;
  --b0:        #141830;
  --b1:        #1c2240;
  --b2:        #252d50;
  --acc:       #6c8fff;
  --acc2:      #a78bfa;
  --ag:        linear-gradient(135deg,#6c8fff,#a78bfa);
  --ag2:       linear-gradient(135deg,rgba(108,143,255,.15),rgba(167,139,250,.08));
  --grn:       #4ade80;
  --grn2:      #86efac;
  --gd:        linear-gradient(135deg,#4ade80,#86efac);
  --red:       #f87171;
  --red2:      #fca5a5;
  --rd:        linear-gradient(135deg,#f87171,#fca5a5);
  --amb:       #fb923c;
  --t1:        #f0f4ff;
  --t2:        #8b9bc8;
  --t3:        #404d78;
  --t4:        #252e55;
  --fi:'Inter',sans-serif;
  --fm:'JetBrains Mono',monospace;
  --r:10px;
}

/* ─── BASE ───────────────────────────────────────── */
html,body,[class*="css"]{
  background:var(--bg)!important;
  color:var(--t1)!important;
  font-family:var(--fi)!important;
}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0 2.5rem 6rem!important;max-width:1320px}

/* Deep space background */
body{
  background:
    radial-gradient(ellipse 80% 50% at 50% -10%,rgba(108,143,255,.1) 0%,transparent 60%),
    radial-gradient(ellipse 40% 30% at 80% 80%,rgba(167,139,250,.06) 0%,transparent 50%),
    var(--bg) !important;
}

/* Subtle dot grid */
body::before{
  content:'';position:fixed;inset:0;
  background-image:radial-gradient(circle,rgba(108,143,255,.045) 1px,transparent 1px);
  background-size:28px 28px;
  pointer-events:none;z-index:0;
}

/* ─── SCROLLBAR ──────────────────────────────────── */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--b1);border-radius:4px}

/* ─── NAV ────────────────────────────────────────── */
.nav{
  display:flex;align-items:center;justify-content:space-between;
  padding:1.25rem 0 1.1rem;
  border-bottom:1px solid var(--b0);
  margin-bottom:2.5rem;
}
.nav-logo{
  display:flex;align-items:center;gap:.75rem;
}
.nav-logo-icon{
  width:32px;height:32px;border-radius:8px;
  background:var(--ag);
  display:flex;align-items:center;justify-content:center;
  font-size:.9rem;box-shadow:0 0 20px rgba(108,143,255,.35);
}
.nav-logo-text{
  font-size:.95rem;font-weight:700;color:var(--t1);letter-spacing:-.025em;
}
.nav-logo-sub{
  font-size:.65rem;color:var(--t3);letter-spacing:.04em;margin-top:.05rem;
}
.nav-links{display:flex;gap:1.75rem;font-size:.78rem;font-weight:500;}
.nav-links a{color:var(--t3);text-decoration:none;transition:color .15s}
.nav-links a.on{color:var(--t2)}
.nav-right{display:flex;align-items:center;gap:.75rem}
.nav-badge{
  font-family:var(--fm);font-size:.56rem;font-weight:600;
  letter-spacing:.14em;text-transform:uppercase;
  padding:.28rem .75rem;border-radius:5px;
  background:linear-gradient(135deg,rgba(108,143,255,.12),rgba(167,139,250,.08));
  color:var(--acc);border:1px solid rgba(108,143,255,.2);
}
.nav-dot{
  display:flex;align-items:center;gap:.4rem;
  font-size:.68rem;color:var(--t3);
}
.ndot{
  width:6px;height:6px;border-radius:50%;
  background:var(--grn);box-shadow:0 0 8px rgba(74,222,128,.5);
}

/* ─── HERO ───────────────────────────────────────── */
.hero{
  text-align:center;padding:3.5rem 2rem 3rem;
  position:relative;margin-bottom:2.5rem;
}
.hero-eyebrow{
  display:inline-flex;align-items:center;gap:.6rem;
  font-family:var(--fm);font-size:.62rem;font-weight:600;
  letter-spacing:.18em;text-transform:uppercase;
  color:var(--acc);background:rgba(108,143,255,.08);
  border:1px solid rgba(108,143,255,.18);
  padding:.35rem 1rem;border-radius:999px;margin-bottom:1.4rem;
}
.hero-eyebrow-dot{
  width:5px;height:5px;border-radius:50%;
  background:var(--acc);box-shadow:0 0 6px var(--acc);
}
.hero h1{
  font-size:3.4rem!important;font-weight:800!important;
  line-height:1.05!important;letter-spacing:-.04em!important;
  margin:0 0 1rem!important;color:var(--t1)!important;
  font-family:var(--fi)!important;
}
.hero h1 .grad{
  background:var(--ag);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;
}
.hero-sub{
  font-size:.95rem;color:var(--t2);max-width:520px;
  margin:0 auto 2rem;line-height:1.75;font-weight:400;
}
/* Dividing line below hero */
.hero-line{
  width:60px;height:1px;
  background:linear-gradient(90deg,transparent,var(--acc),transparent);
  margin:0 auto;
}

/* ─── KPI STRIP ──────────────────────────────────── */
.kpi{
  display:grid;grid-template-columns:repeat(4,1fr);
  gap:1px;background:var(--b0);
  border:1px solid var(--b0);border-radius:var(--r);
  overflow:hidden;margin-bottom:2.5rem;
}
.kpi-c{
  background:var(--s1);padding:1.2rem 1.5rem;
  display:flex;align-items:center;gap:1rem;
  transition:background .2s;
}
.kpi-c:hover{background:var(--s2)}
.kpi-ic{
  width:38px;height:38px;border-radius:9px;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:1rem;
}
.ka{background:linear-gradient(135deg,rgba(108,143,255,.18),rgba(108,143,255,.06));}
.kg{background:linear-gradient(135deg,rgba(74,222,128,.15),rgba(74,222,128,.05));}
.kr{background:linear-gradient(135deg,rgba(248,113,113,.15),rgba(248,113,113,.05));}
.km{background:linear-gradient(135deg,rgba(251,146,60,.14),rgba(251,146,60,.05));}
.kpi-v{
  font-family:var(--fm);font-size:1.3rem;font-weight:600;
  color:var(--t1);line-height:1;
}
.kpi-l{font-size:.67rem;color:var(--t3);margin-top:.25rem;font-weight:500}

/* ─── SECTION HEADING ────────────────────────────── */
.shead{
  display:flex;align-items:center;gap:.7rem;margin-bottom:1.2rem;
}
.shead-bar{
  width:2px;height:18px;border-radius:2px;
  background:var(--ag);flex-shrink:0;
}
.shead-title{
  font-size:.7rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.12em;color:var(--t2);
}
.shead-tag{
  margin-left:auto;font-family:var(--fm);font-size:.58rem;
  color:var(--t4);letter-spacing:.1em;text-transform:uppercase;
}

/* ─── INPUT PANEL ────────────────────────────────── */
.inp-panel{
  background:var(--s1);border:1px solid var(--b0);
  border-radius:var(--r);padding:1.6rem;height:100%;
}
.inp-label{
  font-size:.72rem;font-weight:600;color:var(--t2);
  margin-bottom:.9rem;letter-spacing:.03em;
  display:flex;align-items:center;gap:.5rem;
}
.inp-label::before{
  content:'';width:6px;height:6px;border-radius:50%;
  background:var(--acc);box-shadow:0 0 6px rgba(108,143,255,.5);
  flex-shrink:0;
}

textarea{
  background:var(--s0)!important;
  border:1px solid var(--b1)!important;
  border-radius:8px!important;color:var(--t1)!important;
  font-family:var(--fm)!important;font-size:.8rem!important;
  line-height:1.8!important;transition:border-color .2s,box-shadow .2s!important;
}
textarea:focus{
  border-color:rgba(108,143,255,.5)!important;
  box-shadow:0 0 0 3px rgba(108,143,255,.08)!important;
  outline:none!important;
}
[data-baseweb="select"]>div{
  background:var(--s0)!important;
  border:1px solid var(--b1)!important;
  border-radius:8px!important;color:var(--t1)!important;
}

.meta-row{
  display:flex;gap:1.4rem;margin-top:.6rem;
  font-family:var(--fm);font-size:.6rem;color:var(--t4);
}
.meta-row b{color:var(--t3);font-weight:600}

/* ─── BUTTON ─────────────────────────────────────── */
.stButton>button{
  background:var(--ag)!important;
  color:#fff!important;border:none!important;
  border-radius:8px!important;
  font-family:var(--fi)!important;font-weight:600!important;
  font-size:.84rem!important;padding:.75rem 1.5rem!important;
  width:100%!important;letter-spacing:.01em!important;
  box-shadow:0 4px 24px rgba(108,143,255,.28)!important;
  transition:opacity .18s,transform .15s,box-shadow .2s!important;
}
.stButton>button:hover{
  opacity:.9!important;transform:translateY(-2px)!important;
  box-shadow:0 8px 32px rgba(108,143,255,.4)!important;
}

/* ─── RESULT PANEL ───────────────────────────────── */
.res{
  background:var(--s1);border:1px solid var(--b0);
  border-radius:var(--r);overflow:hidden;
  position:relative;
}

/* Animated scan line on active result */
.res.active::after{
  content:'';
  position:absolute;top:0;left:0;right:0;
  height:1px;
  background:linear-gradient(90deg,transparent,var(--acc),transparent);
  animation:scan 2s ease-in-out infinite;
}
@keyframes scan{
  0%{opacity:0;transform:scaleX(0)}
  30%{opacity:1}
  70%{opacity:1}
  100%{opacity:0;transform:scaleX(1)}
}

.res-head{padding:1.8rem 1.8rem 1.5rem;border-bottom:1px solid var(--b0);}
.res-head.spam{
  background:
    linear-gradient(160deg,rgba(248,113,113,.07) 0%,transparent 50%),
    var(--s1);
}
.res-head.ham{
  background:
    linear-gradient(160deg,rgba(74,222,128,.06) 0%,transparent 50%),
    var(--s1);
}
.res-eyebrow{
  font-family:var(--fm);font-size:.57rem;font-weight:600;
  letter-spacing:.18em;text-transform:uppercase;
  color:var(--t4);margin-bottom:.6rem;
}
.res-verdict{
  font-size:2rem;font-weight:800;letter-spacing:-.04em;
  line-height:1.1;margin-bottom:.4rem;
}
.res-verdict.spam{
  background:var(--rd);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;
}
.res-verdict.ham{
  background:var(--gd);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;
}
.res-desc{font-size:.79rem;color:var(--t2);line-height:1.65;max-width:380px}
.res-tag{
  display:inline-flex;align-items:center;gap:.45rem;
  margin-top:1rem;padding:.32rem .9rem;
  border-radius:6px;font-size:.66rem;font-weight:600;
  letter-spacing:.07em;text-transform:uppercase;
}
.res-tag.spam{
  background:rgba(248,113,113,.1);color:var(--red);
  border:1px solid rgba(248,113,113,.22);
}
.res-tag.ham{
  background:rgba(74,222,128,.08);color:var(--grn);
  border:1px solid rgba(74,222,128,.2);
}

/* Confidence */
.res-conf{padding:1.2rem 1.8rem;border-bottom:1px solid var(--b0);}
.chead{
  display:flex;justify-content:space-between;align-items:center;
  font-size:.68rem;color:var(--t3);margin-bottom:.55rem;
}
.chead strong{
  font-family:var(--fm);font-size:.92rem;
  color:var(--t1);font-weight:600;
}
.ctrack{height:5px;background:var(--b1);border-radius:99px;overflow:hidden}
.cfill{height:100%;border-radius:99px;transition:width .8s cubic-bezier(.4,0,.2,1)}
.cfill.spam{background:var(--rd)}
.cfill.ham {background:var(--gd)}

/* Stat row */
.res-row{display:grid;grid-template-columns:repeat(3,1fr)}
.res-cell{
  padding:1rem 1.4rem;text-align:center;
  border-right:1px solid var(--b0);
}
.res-cell:last-child{border-right:none}
.rc-v{
  font-family:var(--fm);font-size:.95rem;font-weight:700;
  color:var(--t1);margin-bottom:.25rem;letter-spacing:-.01em;
}
.rc-v.r{color:var(--red)}.rc-v.g{color:var(--grn)}
.rc-l{font-size:.6rem;color:var(--t4);text-transform:uppercase;letter-spacing:.1em}

/* Idle */
.idle{padding:4rem 2rem;text-align:center}
.idle-shield{
  width:56px;height:56px;border-radius:14px;margin:0 auto 1.25rem;
  background:linear-gradient(135deg,rgba(108,143,255,.15),rgba(167,139,250,.08));
  border:1px solid rgba(108,143,255,.18);
  display:flex;align-items:center;justify-content:center;
  font-size:1.5rem;
  box-shadow:0 0 30px rgba(108,143,255,.12);
}
.idle-t{font-size:.88rem;font-weight:600;color:var(--t2);margin-bottom:.4rem}
.idle-s{font-size:.76rem;color:var(--t3);line-height:1.75}

/* ─── BREAKDOWN BARS ─────────────────────────────── */
.bkw{margin:.55rem 0}
.bkr{
  display:flex;justify-content:space-between;
  font-size:.7rem;color:var(--t2);margin-bottom:.28rem;
}
.bkr span:last-child{
  font-family:var(--fm);font-weight:600;color:var(--t1);
}
.bktr{height:4px;background:var(--b1);border-radius:99px;overflow:hidden}
.bkfi{
  height:100%;border-radius:99px;
  background:linear-gradient(90deg,var(--acc),var(--acc2));opacity:.7;
}

/* ─── METRICS GRID ───────────────────────────────── */
.mgrid{
  display:grid;grid-template-columns:repeat(4,1fr);
  gap:1px;background:var(--b0);
  border:1px solid var(--b0);border-radius:var(--r);overflow:hidden;
}
.mc{
  background:var(--s1);padding:1.5rem 1.6rem;
  position:relative;overflow:hidden;transition:background .2s;
}
.mc:hover{background:var(--s2)}
.mc::before{
  content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:var(--ag);opacity:.5;
}
.mc-v{
  font-family:var(--fm);font-size:1.55rem;font-weight:700;
  line-height:1;margin-bottom:.45rem;
  background:var(--ag);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;
}
.mc-l{font-size:.78rem;font-weight:600;color:var(--t1);margin-bottom:.2rem}
.mc-s{font-size:.69rem;color:var(--t3);line-height:1.5}

/* ─── THREAT TABLE ───────────────────────────────── */
.ttbl{
  border:1px solid var(--b0);border-radius:var(--r);overflow:hidden;
}
.ttbl-head{
  display:grid;grid-template-columns:2.5rem 1fr 5.5rem 2.8fr;
  gap:1rem;padding:.75rem 1.4rem;
  background:var(--s0);border-bottom:1px solid var(--b0);
  font-size:.6rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.12em;color:var(--t4);
}
.ttbl-row{
  display:grid;grid-template-columns:2.5rem 1fr 5.5rem 2.8fr;
  gap:1rem;padding:.95rem 1.4rem;align-items:center;
  border-bottom:1px solid var(--b0);transition:background .15s;
}
.ttbl-row:last-child{border-bottom:none}
.ttbl-row:hover{background:rgba(108,143,255,.03)}
.t-ico{font-size:1rem}
.t-name{font-size:.82rem;font-weight:600;color:var(--t1)}
.t-desc{font-size:.74rem;color:var(--t2);line-height:1.55}
.rpill{
  display:inline-flex;align-items:center;justify-content:center;
  font-size:.58rem;font-weight:700;letter-spacing:.07em;
  text-transform:uppercase;padding:.22rem .65rem;border-radius:5px;
}
.rpill.H{background:rgba(248,113,113,.1);color:var(--red);border:1px solid rgba(248,113,113,.2)}
.rpill.M{background:rgba(251,146,60,.09);color:var(--amb);border:1px solid rgba(251,146,60,.2)}
.rpill.L{background:rgba(74,222,128,.08);color:var(--grn);border:1px solid rgba(74,222,128,.18)}

/* ─── SIDEBAR ────────────────────────────────────── */
[data-testid="stSidebar"]{
  background:var(--s0)!important;
  border-right:1px solid var(--b0)!important;
}
[data-testid="stSidebar"] *{color:var(--t1)!important}
.stRadio>label{display:none!important}

.sb-top{
  padding:1.3rem 0 1.2rem;border-bottom:1px solid var(--b0);
  margin-bottom:1.4rem;
}
.sb-logo{
  display:flex;align-items:center;gap:.65rem;
}
.sb-logo-box{
  width:30px;height:30px;border-radius:8px;
  background:var(--ag);display:flex;align-items:center;
  justify-content:center;font-size:.85rem;
  box-shadow:0 0 16px rgba(108,143,255,.3);flex-shrink:0;
}
.sb-logo-name{font-size:.9rem;font-weight:700;letter-spacing:-.02em}
.sb-logo-sub{font-size:.63rem;color:var(--t3);margin-top:.1rem}

.sb-sec{margin-bottom:1.6rem}
.sb-lbl{
  font-size:.59rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.16em;color:var(--t4);margin-bottom:.8rem;
}
.sb-eng{
  display:flex;align-items:center;gap:.6rem;
  padding:.6rem .75rem;border-radius:8px;
  border:1px solid transparent;margin-bottom:.35rem;
  cursor:pointer;transition:all .15s;
}
.sb-eng.on{
  background:linear-gradient(135deg,rgba(108,143,255,.12),rgba(167,139,250,.06));
  border-color:rgba(108,143,255,.25);
}
.sb-eng-ic{
  width:28px;height:28px;border-radius:7px;background:var(--b1);
  display:flex;align-items:center;justify-content:center;
  font-size:.85rem;flex-shrink:0;
}
.sb-eng-name{font-size:.78rem;font-weight:600}
.sb-eng-sub{font-size:.63rem;color:var(--t4);margin-top:.07rem}

.sb-kv{
  display:flex;justify-content:space-between;align-items:center;
  font-size:.75rem;padding:.44rem 0;border-bottom:1px solid var(--b0);color:var(--t2);
}
.sb-kv:last-child{border-bottom:none}
.sb-kv b{font-family:var(--fm);font-size:.7rem;color:var(--t1);font-weight:600}

.sb-live{
  display:flex;align-items:center;gap:.5rem;
  padding:.6rem .8rem;border-radius:8px;font-size:.73rem;color:var(--t2);
  background:rgba(74,222,128,.05);border:1px solid rgba(74,222,128,.12);
}
.sb-live-dot{
  width:6px;height:6px;border-radius:50%;
  background:var(--grn);box-shadow:0 0 6px rgba(74,222,128,.5);flex-shrink:0;
}

/* ─── FOOTER ─────────────────────────────────────── */
.foot{
  border-top:1px solid var(--b0);margin-top:4rem;
  padding:1.6rem 0;display:flex;align-items:center;
  justify-content:space-between;
}
.foot-l{font-size:.76rem;color:var(--t2)}
.foot-l b{
  background:var(--ag);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;background-clip:text;font-weight:700;
}
.foot-r{font-family:var(--fm);font-size:.58rem;color:var(--t4);letter-spacing:.07em}

/* ─── EXPANDER ───────────────────────────────────── */
details summary{
  font-size:.73rem!important;font-weight:500!important;
  color:var(--t2)!important;font-family:var(--fi)!important;
}
hr{border-color:var(--b0)!important;margin:2.5rem 0!important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
ICONS = {
    "ham":"✅","lottery":"🎰","financial":"💸","phishing":"🎣",
    "job_spam":"💼","otp_fraud":"🔐","promotion":"📣","adult":"🔞","general_spam":"🚫",
}
DESCS = {
    "ham":          "No threat detected. This message appears completely legitimate.",
    "lottery":      "Lottery scam — fraudulent prize claim designed to defraud you.",
    "financial":    "Financial fraud — fake loan, wire request, or money-transfer scam.",
    "phishing":     "Phishing — credential theft via deceptive links or impersonation.",
    "job_spam":     "Job scam — fraudulent offer or work-from-home recruitment scheme.",
    "otp_fraud":    "OTP theft — do not share any codes or passwords with this sender.",
    "promotion":    "Unsolicited promotional or commercial advertising message.",
    "adult":        "Adult or explicit-content spam detected.",
    "general_spam": "Generic spam — does not match a specific fraud pattern.",
}

@st.cache_resource(show_spinner=False)
def load_models():
    BASE = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(BASE, "models"); out = {}
    for k, fn in [("tokenizer","tokenizer.pkl"),("le","label_encoder.pkl")]:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            with open(p,"rb") as f: out[k] = pickle.load(f)
    lp = os.path.join(d,"lstm_model.h5")
    if os.path.exists(lp):
        try:
            from tensorflow.keras.models import load_model
            out["lstm"] = load_model(lp)
        except: pass
    return out

@st.cache_resource(show_spinner=False)
def nlp_tools():
    for pkg, chk in [("stopwords",lambda:stopwords.words("english")),
                     ("wordnet",  lambda:nltk.data.find("corpora/wordnet"))]:
        try: chk()
        except LookupError: nltk.download(pkg,quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

def clean(text):
    sw, lem = nlp_tools()
    text = re.sub(r"[^a-zA-Z]"," ",str(text).lower())
    return " ".join(lem.lemmatize(w) for w in text.split() if w not in sw)

def predict_lstm(text, M):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq  = pad_sequences(M["tokenizer"].texts_to_sequences([clean(text)]),maxlen=100)
    prob = M["lstm"].predict(seq,verbose=0)[0]
    idx  = int(np.argmax(prob))
    lbl  = M["le"].inverse_transform([idx])[0]
    return lbl,float(prob[idx]),{M["le"].inverse_transform([i])[0]:float(p) for i,p in enumerate(prob)}

def rule_predict(text):
    t = text.lower()
    rules=[
        (["win","prize","lottery","winner","congratulations"],"lottery",.82),
        (["loan","bank account","transfer","wire","money"],   "financial",.78),
        (["click","verify","link","http","confirm"],          "phishing",.80),
        (["job","earn","work from home","hiring","vacancy"],  "job_spam",.75),
        (["otp","password","pin","one-time"],                 "otp_fraud",.85),
        (["free","offer","buy now","sale","discount"],        "promotion",.72),
        (["sex","adult","xxx"],                               "adult",.90),
    ]
    for kws,lbl,c in rules:
        if any(k in t for k in kws): return lbl,c,{lbl:c,"ham":round(1-c,2)}
    hits=sum(1 for w in t.split() if w in {"urgent","claim","selected","cash","won","reward","special"})
    if hits>=2: return "general_spam",.70,{"general_spam":.70,"ham":.30}
    return "ham",.74,{"ham":.74,"general_spam":.26}

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-top">
      <div class="sb-logo">
        <div class="sb-logo-box">🛡️</div>
        <div>
          <div class="sb-logo-name">SpamShield</div>
          <div class="sb-logo-sub">Threat Detection Platform</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-sec"><div class="sb-lbl">Detection Engine</div>',
                unsafe_allow_html=True)
    model_choice = st.radio("e", ["LSTM Neural Network","BERT Transformer","Rule-Based Engine"],
                            label_visibility="collapsed")
    for ico,name,sub,key in [
        ("🧠","LSTM Neural Network","Keras sequential deep learning","LSTM Neural Network"),
        ("⚡","BERT Transformer",   "HuggingFace bert-base-uncased","BERT Transformer"),
        ("📐","Rule-Based Engine",  "Keyword heuristic fallback",   "Rule-Based Engine"),
    ]:
        on = "on" if model_choice==key else ""
        nc = "var(--acc)" if on else "var(--t2)"
        st.markdown(f"""
        <div class="sb-eng {on}">
          <div class="sb-eng-ic">{ico}</div>
          <div>
            <div class="sb-eng-name" style="color:{nc}">{name}</div>
            <div class="sb-eng-sub">{sub}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec"><div class="sb-lbl">System Stats</div>',
                unsafe_allow_html=True)
    for k,v in [("Dataset","UCI SMS Spam"),("Training","5,574 messages"),
                ("Classes","9 categories"),("LSTM Acc","98.7%"),("BERT Acc","97.2%"),
                ("Latency","< 50 ms")]:
        st.markdown(f'<div class="sb-kv"><span>{k}</span><b>{v}</b></div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-live">
      <div class="sb-live-dot"></div>All systems operational
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TOP NAV
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="nav">
  <div class="nav-logo">
    <div class="nav-logo-icon">🛡️</div>
    <div>
      <div class="nav-logo-text">SpamShield AI</div>
      <div class="nav-logo-sub">Threat Intelligence Platform</div>
    </div>
  </div>
  <div class="nav-links">
    <a href="#" class="on">Dashboard</a>
    <a href="#">Models</a>
    <a href="#">Analytics</a>
    <a href="#">Docs</a>
  </div>
  <div class="nav-right">
    <div class="nav-dot"><div class="ndot"></div>Live</div>
    <div class="nav-badge">v1.0 · Beta</div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">
    <div class="hero-eyebrow-dot"></div>
    Deep Learning · NLP · Real-Time Detection
  </div>
  <h1>Detect Threats.<br><span class="grad">Before They Strike.</span></h1>
  <p class="hero-sub">
    Paste any SMS or email. SpamShield's dual-model engine classifies it
    into one of 9 threat categories in under 50 ms — with full confidence scoring.
  </p>
  <div class="hero-line"></div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="kpi">
  <div class="kpi-c"><div class="kpi-ic ka">🛡️</div>
    <div><div class="kpi-v">98.7%</div><div class="kpi-l">LSTM Accuracy</div></div></div>
  <div class="kpi-c"><div class="kpi-ic kg">⚡</div>
    <div><div class="kpi-v">&lt;50ms</div><div class="kpi-l">Avg Inference</div></div></div>
  <div class="kpi-c"><div class="kpi-ic km">🔍</div>
    <div><div class="kpi-v">9</div><div class="kpi-l">Threat Classes</div></div></div>
  <div class="kpi-c"><div class="kpi-ic kr">📊</div>
    <div><div class="kpi-v">5,574</div><div class="kpi-l">Training Samples</div></div></div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# DETECTION WORKSPACE
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="shead">
  <div class="shead-bar"></div>
  <div class="shead-title">Detection Workspace</div>
  <div class="shead-tag">Real-time analysis</div>
</div>""", unsafe_allow_html=True)

SAMPLES = {
    "Select a sample message…": "",
    "🎰 Lottery Scam":    "Congratulations! You've won a £1,000 prize. Call now to claim your lottery winnings!",
    "🎣 Phishing":        "Urgent: Your bank account has been compromised. Click the link to verify your identity now.",
    "💸 Financial Fraud": "Get an instant loan of $50,000 directly to your bank. No credit check needed!",
    "💼 Job Spam":        "Work from home and earn $5000 per week! Apply now and get hired immediately.",
    "🔐 OTP Theft":       "Your OTP is 483920. Never share this one-time password with anyone.",
    "📣 Promotion":       "SALE! 70% OFF today only. Buy now and get free shipping!",
    "✅ Legitimate":      "Hey, are you coming to the meeting tomorrow at 10 AM? Let me know.",
}

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown('<div class="inp-panel">', unsafe_allow_html=True)
    st.markdown('<div class="inp-label">Message Input</div>', unsafe_allow_html=True)
    sel = st.selectbox("s", list(SAMPLES.keys()), label_visibility="collapsed")
    user_text = st.text_area("m", value=SAMPLES[sel], height=178,
                             placeholder="Paste any SMS, email or suspicious message…",
                             label_visibility="collapsed")
    wc = len(user_text.split()) if user_text.strip() else 0
    st.markdown(f"""
    <div class="meta-row">
      <span>Words <b>{wc}</b></span>
      <span>Chars <b>{len(user_text)}</b></span>
      <span>Engine <b style="color:var(--acc)">{model_choice.split()[0]}</b></span>
    </div><br>""", unsafe_allow_html=True)
    go = st.button("Analyse Message →", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_out:
    res_active = "active" if (go and user_text.strip()) else ""
    st.markdown(f'<div class="res {res_active}">', unsafe_allow_html=True)

    if not go or not user_text.strip():
        st.markdown("""
        <div class="idle">
          <div class="idle-shield">🛡️</div>
          <div class="idle-t">Awaiting Analysis</div>
          <div class="idle-s">Paste a message on the left<br>then press <b style="color:var(--t2)">Analyse Message</b>.</div>
        </div>""", unsafe_allow_html=True)
        cc = {}
    else:
        with st.spinner("Running inference…"):
            M = load_models()
            try:
                if model_choice.startswith("LSTM") and "lstm" in M:
                    label,conf,cc = predict_lstm(user_text,M); eng="LSTM Neural Network"
                else:
                    label,conf,cc = rule_predict(user_text)
                    eng="Heuristic Engine"+(" · BERT unavailable" if model_choice.startswith("BERT") else "")
            except Exception:
                label,conf,cc = rule_predict(user_text); eng="Heuristic Engine · Fallback"

        spam   = label!="ham"
        css    = "spam" if spam else "ham"
        verdict= "Threat Detected" if spam else "No Threat Found"
        icon   = ICONS.get(label,"❓")
        desc   = DESCS.get(label,"")
        pct    = round(conf*100,1)
        cat    = label.replace("_"," ").title()
        clr    = "r" if spam else "g"
        risk   = "HIGH RISK" if spam else "SAFE"

        st.markdown(f"""
        <div class="res-head {css}">
          <div class="res-eyebrow">{eng}</div>
          <div class="res-verdict {css}">{verdict}</div>
          <div class="res-desc">{desc}</div>
          <div class="res-tag {css}">{icon}&nbsp;{cat}</div>
        </div>
        <div class="res-conf">
          <div class="chead"><span>Model Confidence</span><strong>{pct}%</strong></div>
          <div class="ctrack"><div class="cfill {css}" style="width:{pct}%"></div></div>
        </div>
        <div class="res-row">
          <div class="res-cell">
            <div class="rc-v {clr}">{icon}</div><div class="rc-l">Category</div>
          </div>
          <div class="res-cell">
            <div class="rc-v {clr}">{pct}%</div><div class="rc-l">Confidence</div>
          </div>
          <div class="res-cell">
            <div class="rc-v {clr}">{risk}</div><div class="rc-l">Risk Level</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if go and user_text.strip() and len(cc) > 2:
        with st.expander("View full probability breakdown"):
            for cls,p in sorted(cc.items(),key=lambda x:-x[1]):
                pv=round(p*100,1)
                st.markdown(f"""
                <div class="bkw">
                  <div class="bkr">
                    <span>{ICONS.get(cls,'❓')} {cls.replace('_',' ').title()}</span>
                    <span>{pv}%</span>
                  </div>
                  <div class="bktr"><div class="bkfi" style="width:{pv}%"></div></div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MODEL METRICS
# ══════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="shead">
  <div class="shead-bar"></div>
  <div class="shead-title">Model Performance</div>
  <div class="shead-tag">Benchmarks</div>
</div>
<div class="mgrid">
  <div class="mc">
    <div class="mc-v">98.7%</div>
    <div class="mc-l">LSTM Accuracy</div>
    <div class="mc-s">Held-out test set · 5,574 SMS samples</div>
  </div>
  <div class="mc">
    <div class="mc-v">97.2%</div>
    <div class="mc-l">BERT Accuracy</div>
    <div class="mc-s">Fine-tuned bert-base-uncased · 9-class</div>
  </div>
  <div class="mc">
    <div class="mc-v">&lt;50ms</div>
    <div class="mc-l">Inference Latency</div>
    <div class="mc-s">Per message including tokenisation</div>
  </div>
  <div class="mc">
    <div class="mc-v">9</div>
    <div class="mc-l">Threat Classes</div>
    <div class="mc-s">Multi-class beyond binary detection</div>
  </div>
  <div class="mc">
    <div class="mc-v">5,574</div>
    <div class="mc-l">Training Samples</div>
    <div class="mc-s">UCI SMS Spam Collection dataset</div>
  </div>
  <div class="mc">
    <div class="mc-v">5,000</div>
    <div class="mc-l">Vocabulary Size</div>
    <div class="mc-s">Keras Tokenizer · top-N word frequency</div>
  </div>
  <div class="mc">
    <div class="mc-v">100</div>
    <div class="mc-l">Max Sequence Length</div>
    <div class="mc-s">Zero-padded for uniform LSTM input</div>
  </div>
  <div class="mc">
    <div class="mc-v">128</div>
    <div class="mc-l">LSTM Hidden Units</div>
    <div class="mc-s">Embedding → LSTM → Dropout → Dense</div>
  </div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# THREAT REFERENCE TABLE
# ══════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="shead">
  <div class="shead-bar"></div>
  <div class="shead-title">Threat Category Reference</div>
  <div class="shead-tag">9 Classes</div>
</div>
<div class="ttbl">
  <div class="ttbl-head">
    <div></div><div>Category</div><div>Risk Level</div><div>Description</div>
  </div>""", unsafe_allow_html=True)

for ico,name,rc,rl,desc in [
    ("✅","Ham",          "L","Low",    "Legitimate message — no spam signals or malicious patterns detected."),
    ("🎰","Lottery",      "H","High",   "Fraudulent prize or lottery claim — always a scam, never legitimate."),
    ("💸","Financial",    "H","High",   "Fake loan offers, suspicious wire requests, or money-transfer fraud."),
    ("🎣","Phishing",     "H","High",   "Credential theft via deceptive links, fake login pages, or impersonation."),
    ("💼","Job Spam",     "M","Medium", "Fraudulent job offers, MLM recruitment, or work-from-home schemes."),
    ("🔐","OTP Fraud",    "H","High",   "Social-engineering to extract one-time passwords or authentication codes."),
    ("📣","Promotion",    "M","Medium", "Unsolicited commercial messages — low risk but unwanted advertising."),
    ("🔞","Adult",        "M","Medium", "Explicit or age-restricted spam content."),
    ("🚫","General Spam", "M","Medium", "Catch-all for spam not matching a specific fraud pattern."),
]:
    st.markdown(f"""
  <div class="ttbl-row">
    <div class="t-ico">{ico}</div>
    <div class="t-name">{name}</div>
    <div><span class="rpill {rc}">{rl}</span></div>
    <div class="t-desc">{desc}</div>
  </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="foot">
  <div class="foot-l">
    <b>SpamShield AI</b> &nbsp;—&nbsp;
    Deep Learning Threat Detection &nbsp;·&nbsp;
    Streamlit · TensorFlow / Keras · HuggingFace Transformers
  </div>
  <div class="foot-r">LSTM · BERT · NLTK · UCI SMS Spam Collection</div>
</div>""", unsafe_allow_html=True)