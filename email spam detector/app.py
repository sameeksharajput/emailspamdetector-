"""
Email Spam Detection - Streamlit app (redesigned UI)

Run:
    python -m streamlit run app.py

Needs (created by train_model.py):
    dataset/spam_model.pkl
    dataset/tfidf_vectorizer.pkl
"""
import html
import os

import joblib
import streamlit as st

MODEL_PATH = "dataset/spam_model.pkl"
VECTORIZER_PATH = "dataset/tfidf_vectorizer.pkl"

EXAMPLES = {
    "Prize scam": "Congratulations! You have won a FREE $1000 gift card. Click the link now to claim your prize before it expires!",
    "Work email": "Hi team, attaching the notes from today's meeting. Please review and share feedback before Friday's sync.",
    "Phishing": "URGENT: Your bank account has been suspended. Verify your password immediately at http://secure-login.example to avoid closure.",
    "Friend text": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
}

st.set_page_config(page_title="MailGuard | Spam Detector", page_icon="🛡️", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 15% 0%, #1b2440 0%, #0b1020 55%, #070a14 100%);
        color: #e6e9f2;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 1150px; }

    .hero { padding: 6px 0 18px 0; }
    .hero h1 {
        font-size: 2.4rem; margin: 0; font-weight: 800;
        background: linear-gradient(90deg, #7aa2ff, #b388ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: #9aa4bf; margin-top: 6px; font-size: 1.02rem; }

    .card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px; padding: 22px 24px;
        backdrop-filter: blur(6px);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 18px; padding: 8px 10px;
    }
    .card h3 { margin-top: 0; font-size: 1.05rem; color: #c9d3f0; letter-spacing: .3px; }

    .stTextArea textarea {
        background: rgba(10,14,30,0.8) !important; color: #e6e9f2 !important;
        border: 1px solid rgba(122,162,255,0.35) !important; border-radius: 12px !important;
        font-size: 1rem !important;
    }
    .stButton > button {
        border-radius: 10px; border: 1px solid rgba(122,162,255,0.4);
        background: rgba(122,162,255,0.10); color: #cfd9ff; font-weight: 600;
    }
    .stButton > button:hover { background: rgba(122,162,255,0.22); border-color: #7aa2ff; color: #fff; }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #5b7cff, #9b6bff); color: white; border: none;
    }

    .verdict { border-radius: 16px; padding: 20px 22px; margin-bottom: 16px; }
    .verdict.spam { background: rgba(255,82,102,0.12); border: 1px solid rgba(255,82,102,0.55); }
    .verdict.ham  { background: rgba(60,214,140,0.12); border: 1px solid rgba(60,214,140,0.55); }
    .verdict .label { font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase; color: #9aa4bf; }
    .verdict .title { font-size: 1.9rem; font-weight: 800; margin: 4px 0 0 0; }
    .verdict.spam .title { color: #ff6b81; }
    .verdict.ham .title { color: #3cd68c; }

    .meter-wrap { margin: 6px 0 4px 0; }
    .meter-track {
        height: 14px; border-radius: 999px; overflow: hidden;
        background: linear-gradient(90deg, #3cd68c 0%, #f5c542 50%, #ff5266 100%);
        position: relative;
    }
    .meter-dot {
        position: absolute; top: -3px; width: 20px; height: 20px; border-radius: 50%;
        background: #fff; border: 3px solid #0b1020; transform: translateX(-50%);
    }
    .meter-scale { display: flex; justify-content: space-between; font-size: 0.75rem; color: #8590ad; margin-top: 6px; }

    .chip {
        display: inline-block; padding: 4px 12px; margin: 3px 4px 3px 0;
        border-radius: 999px; font-size: 0.85rem; font-weight: 600;
    }
    .chip.s { background: rgba(255,82,102,0.18); color: #ff8da0; border: 1px solid rgba(255,82,102,0.4); }
    .chip.h { background: rgba(60,214,140,0.16); color: #7ae8b4; border: 1px solid rgba(60,214,140,0.4); }

    .stat { text-align: center; }
    .stat .n { font-size: 1.7rem; font-weight: 800; color: #cfd9ff; }
    .stat .t { font-size: 0.78rem; color: #8590ad; text-transform: uppercase; letter-spacing: 1px; }

    .hist-row { display: flex; justify-content: space-between; gap: 10px; font-size: 0.88rem;
                padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.07); color: #c3cbe3; }
    .hist-tag { font-weight: 700; }
    .hist-tag.s { color: #ff6b81; } .hist-tag.h { color: #3cd68c; }

    section[data-testid="stSidebar"] { background: #0a0e1c; border-right: 1px solid rgba(255,255,255,0.07); }
    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model loading ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    return model, tfidf


def spam_class_index(model):
    classes = list(model.classes_)
    if "spam" in classes:
        return classes.index("spam")
    if 1 in classes:
        return classes.index(1)
    return len(classes) - 1


def explain(model, tfidf, vec, spam_idx, top_n=6):
    """Return (spam_words, ham_words) that pushed the prediction, if the model is linear."""
    if not hasattr(model, "coef_"):
        return [], []
    coef = model.coef_[0]
    if spam_idx == 0:
        coef = -coef
    names = tfidf.get_feature_names_out()
    row = vec.tocoo()
    contrib = [(names[c], float(v * coef[c])) for c, v in zip(row.col, row.data)]
    spam_words = [w for w, s in sorted(contrib, key=lambda x: -x[1]) if s > 0][:top_n]
    ham_words = [w for w, s in sorted(contrib, key=lambda x: x[1]) if s < 0][:top_n]
    return spam_words, ham_words


def set_message(text):
    st.session_state["message"] = text


# ---------- State ----------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "message" not in st.session_state:
    st.session_state["message"] = ""
if "result" not in st.session_state:
    st.session_state["result"] = None

# ---------- Check model files ----------
if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
    st.error("Model files not found.")
    st.code("python train_model.py --data spam.csv", language="bash")
    st.write("Run the command above once. It creates the files in the `dataset` folder, then refresh this page.")
    st.stop()

model, tfidf = load_artifacts()
spam_idx = spam_class_index(model)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🛡️ MailGuard")
    st.caption("ML-powered spam detector")
    st.markdown("---")
    st.markdown("**How it works**")
    st.write(
        "Your text is converted into TF-IDF features, then a Logistic Regression "
        "classifier estimates how likely it is to be spam."
    )
    st.markdown("---")
    st.markdown("**Recent checks**")
    if not st.session_state["history"]:
        st.caption("Nothing checked yet.")
    for item in reversed(st.session_state["history"][-6:]):
        tag = "SPAM" if item["is_spam"] else "HAM"
        cls = "s" if item["is_spam"] else "h"
        st.markdown(
            f"<div class='hist-row'><span>{html.escape(item['preview'])}</span>"
            f"<span class='hist-tag {cls}'>{tag}</span></div>",
            unsafe_allow_html=True,
        )
    if st.session_state["history"] and st.button("Clear history"):
        st.session_state["history"] = []
        st.session_state["result"] = None
        st.rerun()

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>MailGuard Spam Detector</h1>
        <p>Paste any email or message and get an instant spam verdict with the reasons behind it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 1], gap="large")

# ---------- Left: input ----------
with left:
    with st.container(border=True):
        st.markdown("##### ✉️ Your message")
        st.text_area(
            "Message",
            key="message",
            height=220,
            placeholder="Paste the email or message here...",
            label_visibility="collapsed",
        )

        st.caption("Try an example:")
        cols = st.columns(len(EXAMPLES))
        for col, (name, text) in zip(cols, EXAMPLES.items()):
            col.button(name, on_click=set_message, args=(text,), key=f"ex_{name}")

        analyze = st.button("Analyze message", type="primary")

    if analyze:
        msg = st.session_state["message"].strip()
        if not msg:
            st.warning("Please enter a message first.")
        else:
            vec = tfidf.transform([msg])
            probs = model.predict_proba(vec)[0]
            spam_prob = float(probs[spam_idx]) * 100
            is_spam = spam_prob >= 50
            spam_words, ham_words = explain(model, tfidf, vec, spam_idx)
            st.session_state["result"] = {
                "spam_prob": spam_prob,
                "is_spam": is_spam,
                "spam_words": spam_words,
                "ham_words": ham_words,
                "chars": len(msg),
                "words": len(msg.split()),
            }
            preview = msg if len(msg) <= 34 else msg[:34] + "..."
            st.session_state["history"].append({"preview": preview, "is_spam": is_spam})
            st.rerun()

# ---------- Right: result ----------
with right:
    res = st.session_state["result"]
    if res is None:
        st.markdown(
            "<div class='card' style='text-align:center; padding:60px 24px;'>"
            "<div style='font-size:3rem;'>🔎</div>"
            "<h3 style='margin-top:10px;'>Waiting for a message</h3>"
            "<p style='color:#8590ad;'>Write or pick an example, then press <b>Analyze message</b>.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        cls = "spam" if res["is_spam"] else "ham"
        title = "🚨 Spam detected" if res["is_spam"] else "✅ Looks safe"
        conf = res["spam_prob"] if res["is_spam"] else 100 - res["spam_prob"]

        st.markdown(
            f"""
            <div class="verdict {cls}">
                <div class="label">Verdict</div>
                <div class="title">{title}</div>
                <div style="color:#b7c0da; margin-top:4px;">Model confidence: {conf:.1f}%</div>
            </div>
            <div class="card">
                <h3>Spam likelihood</h3>
                <div class="meter-wrap">
                    <div class="meter-track"><div class="meter-dot" style="left:{res['spam_prob']:.1f}%;"></div></div>
                    <div class="meter-scale"><span>Safe</span><span>{res['spam_prob']:.1f}% spam</span><span>Spam</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='card stat'><div class='n'>{res['spam_prob']:.0f}%</div><div class='t'>Spam score</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='card stat'><div class='n'>{res['words']}</div><div class='t'>Words</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='card stat'><div class='n'>{res['chars']}</div><div class='t'>Characters</div></div>", unsafe_allow_html=True)

        if res["spam_words"] or res["ham_words"]:
            st.write("")
            chips = ""
            if res["spam_words"]:
                chips += "<div style='color:#9aa4bf; font-size:.85rem; margin-bottom:4px;'>Pushed toward spam</div>"
                chips += "".join(f"<span class='chip s'>{html.escape(w)}</span>" for w in res["spam_words"])
            if res["ham_words"]:
                chips += "<div style='color:#9aa4bf; font-size:.85rem; margin:10px 0 4px 0;'>Pushed toward safe</div>"
                chips += "".join(f"<span class='chip h'>{html.escape(w)}</span>" for w in res["ham_words"])
            st.markdown(f"<div class='card'><h3>Why this result?</h3>{chips}</div>", unsafe_allow_html=True)

st.caption("Built for learning purposes. Always double check important emails yourself.")