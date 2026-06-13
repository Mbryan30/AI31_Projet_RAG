import streamlit as st
import requests
import uuid
import time
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG·AI — RGPD",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

RAG_MODES = [
    {"id": "adaptive", "label": "RAG Adaptatif",  "desc": "Pipeline complet avec routing intelligent",   "color": "#6c8fff"},
    {"id": "crag",     "label": "CRAG + Fusion",  "desc": "Corrective RAG avec Reciprocal Rank Fusion",  "color": "#3ddc84"},
    {"id": "hyde",     "label": "HyDE + Rerank",  "desc": "Hypothetical Documents + Cross-encoder",       "color": "#a855f7"},
    {"id": "selfrag",  "label": "Self-RAG",        "desc": "Auto-évaluation ISREL/ISSUP/ISUSE",            "color": "#f5a623"},
]

SUGGESTIONS = [
    ("Droits des personnes",  "Droit d'accès, effacement, portabilité"),
    ("Bases légales",          "Consentement, intérêt légitime, contrat"),
    ("DPO & registre",         "Obligations du délégué à la protection"),
    ("Transferts hors UE",    "Clauses contractuelles types, BCR"),
]

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0a0a0b !important;
    color: #e8e8ea !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
}
[data-testid="stMain"] { background: #0a0a0b !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #111113 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* ── Buttons ── */
.stButton > button {
    background: #6c8fff !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
    width: 100% !important;
}
.stButton > button:hover { background: #4a6ef0 !important; }

/* ── Text input ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #18181c !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    color: #e8e8ea !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(108,143,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(108,143,255,0.08) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #18181c !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    color: #e8e8ea !important;
}

/* ── Radio / Toggle ── */
.stRadio > div { flex-direction: row !important; gap: 8px !important; }
.stRadio label { color: #9090a0 !important; font-size: 12px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #222228; border-radius: 4px; }

/* ── Chat container ── */
.chat-wrap {
    max-width: 760px;
    margin: 0 auto;
    padding: 28px 24px;
}

/* ── Message bubble ── */
.msg-row { display: flex; gap: 12px; margin-bottom: 24px; animation: fadeUp 0.3s ease both; }
.avatar {
    width: 28px; height: 28px; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; flex-shrink: 0; margin-top: 2px;
}
.avatar-user { background: linear-gradient(135deg, #6c8fff, #a855f7); color: white; font-weight: 500; }
.avatar-ai   { background: #18181c; border: 1px solid rgba(255,255,255,0.12); color: #9090a0; }
.msg-body { flex: 1; min-width: 0; }
.msg-header { display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px; }
.msg-name { font-size: 12px; font-weight: 500; color: #e8e8ea; }
.msg-badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    padding: 2px 6px; border-radius: 4px;
    background: rgba(61,220,132,0.1); border: 1px solid rgba(61,220,132,0.2); color: #3ddc84;
}
.msg-time { font-size: 10px; color: #505060; }
.msg-content {
    font-size: 14px; color: #e8e8ea; line-height: 1.65;
}
.msg-content p { margin-bottom: 12px; }
.msg-content p:last-child { margin-bottom: 0; }
.msg-content ul, .msg-content ol { margin-bottom: 12px; padding-left: 20px; }
.msg-content li { margin-bottom: 4px; }
.msg-content strong { font-weight: 500; color: #e8e8ea; }
.msg-content h1,.msg-content h2,.msg-content h3 {
    font-family: 'DM Serif Display', serif;
    color: #e8e8ea; margin: 16px 0 8px;
}
.msg-content h2 { font-size: 16px; }
.msg-content h3 { font-size: 14px; }
.msg-content code {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    background: #18181c; border: 1px solid rgba(255,255,255,0.07);
    padding: 2px 6px; border-radius: 4px; color: #a8d8ff;
}
.msg-content pre {
    background: #18181c; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px; padding: 14px; overflow-x: auto; margin: 12px 0;
}
.msg-content pre code { background: none; border: none; padding: 0; font-size: 12.5px; line-height: 1.6; }

/* ── Sources block ── */
.sources-wrap {
    margin-top: 10px;
    background: #111113;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    overflow: hidden;
}
.sources-header {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px;
    font-size: 11px; color: #9090a0;
}
.sources-header svg { color: #6c8fff; }
.source-item {
    display: flex; align-items: center; gap: 8px;
    background: #18181c; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 7px; padding: 6px 10px; margin: 0 12px 6px;
}
.source-num {
    width: 18px; height: 18px; border-radius: 5px;
    background: rgba(108,143,255,0.15); color: #6c8fff;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.source-title { flex: 1; font-size: 12px; color: #9090a0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.source-score { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #3ddc84; }

/* ── Metrics ── */
.metrics-row { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
.metric-pill {
    display: flex; align-items: center; gap: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #505060;
    padding: 3px 8px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.07);
}
.metric-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }

/* ── Date divider ── */
.date-divider {
    display: flex; align-items: center; gap: 12px; margin: 20px 0;
}
.date-line { flex: 1; height: 1px; background: rgba(255,255,255,0.07); }
.date-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #505060; }

/* ── Welcome screen ── */
.welcome-wrap {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; height: 100%; text-align: center;
    padding: 24px; gap: 16px;
}
.welcome-icon {
    width: 56px; height: 56px; background: #111113;
    border: 1px solid rgba(255,255,255,0.07); border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; margin-bottom: 4px;
}
.welcome-title { font-family: 'DM Serif Display', serif; font-size: 28px; font-weight: 400; color: #e8e8ea; }
.welcome-sub { font-size: 14px; color: #9090a0; max-width: 420px; line-height: 1.6; }
.suggestions-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; max-width: 480px; width: 100%; margin-top: 8px; }
.suggestion-card {
    background: #111113; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px; padding: 14px; text-align: left; cursor: pointer;
    transition: all 0.15s;
}
.suggestion-card:hover { border-color: rgba(255,255,255,0.12); background: #18181c; }
.suggestion-title { font-size: 12.5px; font-weight: 500; color: #e8e8ea; margin-bottom: 2px; }
.suggestion-sub   { font-size: 11px; color: #505060; }

/* ── Input area ── */
.input-area {
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 14px 24px 20px;
    flex-shrink: 0;
}
.input-box {
    max-width: 760px; margin: 0 auto;
    background: #111113; border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px; overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.toolbar {
    display: flex; align-items: center; gap: 6px;
    padding: 10px 12px 0;
}
.mode-pill {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; padding: 4px 8px; border-radius: 6px; border: 1px solid;
    cursor: pointer; transition: all 0.15s;
}
.source-pill {
    display: flex; align-items: center; gap: 4px;
    font-size: 11px; padding: 4px 8px; border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.07); color: #505060;
    cursor: pointer; transition: all 0.15s;
}
.send-btn {
    width: 34px; height: 34px; border-radius: 9px;
    background: #6c8fff; color: white; border: none;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: all 0.15s; flex-shrink: 0;
}
.send-btn:hover { background: #4a6ef0; }
.input-hint {
    display: flex; justify-content: space-between;
    padding: 4px 14px 8px; font-size: 10px; color: #505060;
}

/* ── Topbar ── */
.topbar {
    border-bottom: 1px solid rgba(255,255,255,0.07);
    height: 52px; padding: 0 24px;
    display: flex; align-items: center; gap: 12px;
    flex-shrink: 0;
}
.topbar-title { font-family: 'DM Serif Display', serif; font-size: 16px; color: #e8e8ea; flex: 1; }
.status-pill {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: #9090a0;
    background: #18181c; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 999px; padding: 4px 12px;
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #3ddc84; }

/* ── Sidebar custom ── */
.sidebar-header {
    display: flex; align-items: center; gap: 10px;
    padding: 17px 16px; border-bottom: 1px solid rgba(255,255,255,0.07);
}
.sidebar-logo {
    width: 30px; height: 30px; background: #6c8fff; border-radius: 6px;
    display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0;
}
.sidebar-brand { font-family: 'DM Serif Display', serif; font-size: 17px; color: #e8e8ea; }
.sidebar-badge {
    margin-left: auto; font-size: 9px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase;
    background: rgba(108,143,255,0.15); color: #6c8fff;
    padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(108,143,255,0.25);
}
.sidebar-section {
    padding: 8px 16px 4px;
    font-size: 10px; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: #505060;
}
.session-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px; border-radius: 6px; margin: 0 8px 2px;
    cursor: pointer; transition: all 0.15s;
}
.session-item:hover { background: #18181c; }
.session-item.active { background: rgba(108,143,255,0.12); border: 1px solid rgba(108,143,255,0.2); }
.session-icon { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.session-title { font-size: 12.5px; color: #9090a0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.session-item.active .session-title { color: #e8e8ea; }
.sidebar-stats {
    border-top: 1px solid rgba(255,255,255,0.07);
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
    padding: 10px 14px;
}
.stat-cell { text-align: center; }
.stat-val { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 500; color: #e8e8ea; }
.stat-lbl { font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; color: #505060; }
.sidebar-user {
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 10px 14px; display: flex; align-items: center; gap: 10px;
}
.user-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    background: linear-gradient(135deg, #6c8fff, #a855f7);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 12px; font-weight: 500; flex-shrink: 0;
}
.user-name { font-size: 12.5px; font-weight: 500; color: #e8e8ea; }
.user-sub  { font-size: 10px; color: #505060; }
.online-dot { margin-left: auto; width: 7px; height: 7px; border-radius: 50%; background: #3ddc84; }

/* ── Mode selector pills in sidebar ── */
.mode-selector { padding: 6px 12px 10px; }
.mode-option {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 8px; border-radius: 6px; margin-bottom: 2px;
    cursor: pointer; transition: all 0.15s; border: 1px solid transparent;
}
.mode-option:hover { background: #18181c; }
.mode-option.selected { background: #18181c; }
.mode-dot  { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.mode-label { font-size: 12px; font-weight: 500; color: #e8e8ea; }
.mode-desc  { font-size: 10px; color: #505060; }

/* Streamlit-specific overrides */
.stMarkdown { color: #e8e8ea !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.element-container { margin: 0 !important; }
[data-testid="stChatInput"] {
    background: #111113 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e8e8ea !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
[data-testid="stChatInputSubmitButton"] {
    background: #6c8fff !important;
    border-radius: 9px !important;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes blink {
    0%,100% { opacity: 0.3; }
    50%      { opacity: 1; }
}
.blink { display: inline-block; width: 2px; height: 16px; background: #6c8fff; margin-left: 2px; vertical-align: middle; animation: blink 1.2s ease-in-out infinite; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session_id" not in st.session_state:
    sid = str(uuid.uuid4())
    st.session_state.sessions[sid] = {
        "id": sid, "title": "Nouvelle session",
        "messages": [], "created": time.time(), "updated": time.time(), "mode": "adaptive"
    }
    st.session_state.current_session_id = sid

if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "adaptive"

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Helpers ───────────────────────────────────────────────────────────
def current_session():
    return st.session_state.sessions[st.session_state.current_session_id]

def fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M")

def fmt_date(ts):
    d = datetime.fromtimestamp(ts)
    today = datetime.now().date()
    if d.date() == today:
        return "Aujourd'hui"
    return d.strftime("%d %B %Y")

def new_session():
    sid = str(uuid.uuid4())
    st.session_state.sessions[sid] = {
        "id": sid, "title": "Nouvelle session",
        "messages": [], "created": time.time(), "updated": time.time(), "mode": st.session_state.rag_mode
    }
    st.session_state.current_session_id = sid

def call_api(question: str, session_id: str, mode: str) -> dict:
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "session_id": session_id, "mode": mode, "sources": ["vectorstore"]},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {
            "answer": f"⚠️ Erreur de connexion au backend : {e}",
            "sources": [],
            "metrics": {"latency": 0, "tokens": 0, "strategy": "—"},
        }

def send_message(question: str):
    sess = current_session()
    # User message
    sess["messages"].append({
        "id": str(uuid.uuid4()), "role": "user", "content": question,
        "ts": time.time(), "sources": None, "metrics": None,
    })
    if sess["title"] == "Nouvelle session" and len(sess["messages"]) == 1:
        sess["title"] = question[:40] + ("…" if len(question) > 40 else "")
    sess["updated"] = time.time()

    # API call
    with st.spinner(""):
        result = call_api(question, sess["id"], st.session_state.rag_mode)

    sess["messages"].append({
        "id": str(uuid.uuid4()), "role": "assistant",
        "content": result["answer"],
        "ts": time.time(),
        "sources": result.get("sources", []),
        "metrics": result.get("metrics"),
    })
    sess["updated"] = time.time()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">⬡</div>
        <span class="sidebar-brand">RAG·AI</span>
        <span class="sidebar-badge">RGPD</span>
    </div>
    """, unsafe_allow_html=True)

    # New session button
    st.markdown('<div style="padding:12px 12px 6px;">', unsafe_allow_html=True)
    if st.button("＋  Nouvelle session", key="new_sess"):
        new_session()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Search
    search = st.text_input("", placeholder="🔍  Rechercher…", key="search", label_visibility="collapsed")

    # Mode selector
    st.markdown('<div class="sidebar-section">Mode RAG</div>', unsafe_allow_html=True)
    mode_html = '<div class="mode-selector">'
    for m in RAG_MODES:
        sel = "selected" if st.session_state.rag_mode == m["id"] else ""
        mode_html += f"""
        <div class="mode-option {sel}" style="{'border-color:'+m['color']+'33;' if sel else ''}">
            <div class="mode-dot" style="background:{m['color']}"></div>
            <div>
                <div class="mode-label">{m['label']}</div>
                <div class="mode-desc">{m['desc']}</div>
            </div>
        </div>"""
    mode_html += '</div>'
    st.markdown(mode_html, unsafe_allow_html=True)

    # Mode radio (hidden label, functional)
    mode_labels = [m["label"] for m in RAG_MODES]
    mode_ids    = [m["id"]    for m in RAG_MODES]
    cur_idx = mode_ids.index(st.session_state.rag_mode)
    chosen = st.radio("Mode", mode_labels, index=cur_idx, key="mode_radio", label_visibility="collapsed")
    if mode_ids[mode_labels.index(chosen)] != st.session_state.rag_mode:
        st.session_state.rag_mode = mode_ids[mode_labels.index(chosen)]
        st.rerun()

    # Sessions list
    st.markdown('<div class="sidebar-section">Sessions récentes</div>', unsafe_allow_html=True)
    sessions_sorted = sorted(st.session_state.sessions.values(), key=lambda s: s["updated"], reverse=True)
    filtered = [s for s in sessions_sorted if search.lower() in s["title"].lower()] if search else sessions_sorted

    if not filtered:
        st.markdown('<div style="text-align:center;padding:20px;font-size:12px;color:#505060;">Aucune session</div>', unsafe_allow_html=True)
    else:
        for s in filtered:
            active = s["id"] == st.session_state.current_session_id
            css = "active" if active else ""
            col1, col2 = st.columns([5, 1])
            with col1:
                mode_color = next((m["color"] for m in RAG_MODES if m["id"] == s.get("mode", "adaptive")), "#6c8fff")
                st.markdown(f"""
                <div class="session-item {css}">
                    <div class="session-icon" style="background:{mode_color}"></div>
                    <div class="session-title">{s['title']}</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                if st.button("→", key=f"switch_{s['id']}", help="Ouvrir"):
                    st.session_state.current_session_id = s["id"]
                    st.rerun()

    # Stats
    total_msgs = sum(len(s["messages"]) for s in st.session_state.sessions.values())
    st.markdown(f"""
    <div class="sidebar-stats">
        <div class="stat-cell"><div class="stat-val">{len(st.session_state.sessions)}</div><div class="stat-lbl">Sessions</div></div>
        <div class="stat-cell"><div class="stat-val">{total_msgs}</div><div class="stat-lbl">Messages</div></div>
        <div class="stat-cell"><div class="stat-val">—</div><div class="stat-lbl">Docs RAG</div></div>
    </div>""", unsafe_allow_html=True)

    # User
    st.markdown("""
    <div class="sidebar-user">
        <div class="user-avatar">U</div>
        <div>
            <div class="user-name">Utilisateur</div>
            <div class="user-sub">RAG RGPD · v2</div>
        </div>
        <div class="online-dot"></div>
    </div>""", unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────
sess = current_session()
messages = sess["messages"]

# TopBar
current_mode_cfg = next((m for m in RAG_MODES if m["id"] == st.session_state.rag_mode), RAG_MODES[0])
st.markdown(f"""
<div class="topbar">
    <div class="topbar-title">{sess['title']}</div>
    <div class="status-pill"><div class="status-dot"></div>Pipeline actif</div>
</div>""", unsafe_allow_html=True)

# ── Chat area ─────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not messages:
        # Welcome screen
        suggestions_html = '<div class="suggestions-grid">'
        for title, sub in SUGGESTIONS:
            suggestions_html += f"""
            <div class="suggestion-card" onclick="void(0)">
                <div class="suggestion-title">{title}</div>
                <div class="suggestion-sub">{sub}</div>
            </div>"""
        suggestions_html += '</div>'

        st.markdown(f"""
        <div class="welcome-wrap" style="min-height:60vh;">
            <div class="welcome-icon">⬡</div>
            <div class="welcome-title">RAG·AI — RGPD</div>
            <div class="welcome-sub">Pipeline RAG adaptatif avec CRAG, Self-RAG et indexation hiérarchique RAPTOR. Posez vos questions sur la conformité RGPD.</div>
            {suggestions_html}
        </div>""", unsafe_allow_html=True)

        # Suggestion buttons (functional)
        cols = st.columns(2)
        for i, (title, sub) in enumerate(SUGGESTIONS):
            with cols[i % 2]:
                if st.button(f"**{title}**\n{sub}", key=f"sug_{i}", use_container_width=True):
                    send_message(f"{title} — {sub}")
                    st.rerun()
    else:
        # Group by date
        groups: dict[str, list] = {}
        for msg in messages:
            d = fmt_date(msg["ts"])
            groups.setdefault(d, []).append(msg)

        for date_label, day_msgs in groups.items():
            st.markdown(f"""
            <div class="date-divider chat-wrap" style="padding-bottom:0;padding-top:0;">
                <div class="date-line"></div>
                <span class="date-label">{date_label}</span>
                <div class="date-line"></div>
            </div>""", unsafe_allow_html=True)

            for msg in day_msgs:
                is_user = msg["role"] == "user"
                avatar_cls = "avatar-user" if is_user else "avatar-ai"
                avatar_char = "U" if is_user else "⬡"
                name = "Vous" if is_user else "RAG·AI"
                badge = '' if is_user else '<span class="msg-badge">RAG</span>'
                t = fmt_time(msg["ts"])

                # Render content as markdown via st.markdown inside a custom wrapper
                st.markdown(f"""
                <div class="msg-row chat-wrap" style="padding-bottom:0;">
                    <div class="avatar {avatar_cls}">{avatar_char}</div>
                    <div class="msg-body">
                        <div class="msg-header">
                            <span class="msg-name">{name}</span>
                            {badge}
                            <span class="msg-time">{t}</span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Content rendered as markdown (proper Streamlit rendering)
                with st.container():
                    st.markdown(f'<div class="chat-wrap" style="padding-top:0;padding-bottom:0;"><div class="msg-content" style="padding-left:40px;">', unsafe_allow_html=True)
                    st.markdown(msg["content"])
                    st.markdown('</div></div>', unsafe_allow_html=True)

                # Sources
                if not is_user and msg.get("sources"):
                    sources = msg["sources"]
                    with st.expander(f"📚 {len(sources)} source{'s' if len(sources) > 1 else ''} récupérée{'s' if len(sources) > 1 else ''}", expanded=False):
                        for i, src in enumerate(sources):
                            st.markdown(f"""
                            <div class="source-item">
                                <div class="source-num">{i+1}</div>
                                <div class="source-title" title="{src['title']}">{src['title']}</div>
                                <div class="source-score">{src['score']}%</div>
                            </div>""", unsafe_allow_html=True)

                # Metrics
                if not is_user and msg.get("metrics"):
                    m = msg["metrics"]
                    st.markdown(f"""
                    <div class="metrics-row" style="padding-left:40px;padding-bottom:8px;">
                        <div class="metric-pill"><div class="metric-dot" style="background:#3ddc84"></div>{m['latency']}ms</div>
                        <div class="metric-pill"><div class="metric-dot" style="background:#6c8fff"></div>{m['tokens']} tokens</div>
                        <div class="metric-pill"><div class="metric-dot" style="background:#f5a623"></div>{m['strategy']}</div>
                    </div>""", unsafe_allow_html=True)

# ── Input area ────────────────────────────────────────────────────────
st.markdown(f"""
<div style="border-top:1px solid rgba(255,255,255,0.07);padding:14px 24px 4px;">
    <div style="max-width:760px;margin:0 auto;">
        <div class="toolbar">
            <div class="mode-pill" style="color:{current_mode_cfg['color']};border-color:{current_mode_cfg['color']}55;background:{current_mode_cfg['color']}18;">
                ✦ {current_mode_cfg['label']}
            </div>
            <div class="source-pill">⊞ Sources</div>
            <div class="source-pill">↑ Document</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# Chat input
question = st.chat_input("Posez votre question RGPD…")
if question:
    send_message(question)
    st.rerun()

st.markdown("""
<div style="max-width:760px;margin:0 auto;display:flex;justify-content:space-between;padding:4px 24px 12px;font-size:10px;color:#505060;">
    <span>Entrée pour envoyer</span>
    <span style="font-family:'JetBrains Mono',monospace;">⇧↵ nouvelle ligne</span>
</div>""", unsafe_allow_html=True)
