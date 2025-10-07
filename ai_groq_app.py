"""
groq_streamlit_app.py
Single-file Streamlit app with a modern UI + animations that calls the Groq API.
Features:
 - Paste or read GROQ_API_KEY from env
 - Chat-style interface for generation (choose model)
 - Simple embeddings endpoint example
 - Upload text files (optional) to include as context (basic RAG demo)
 - Lottie animations and CSS for a modern look
Requirements:
 pip install streamlit groq streamlit-lottie python-dotenv
Docs: https://console.groq.com/docs and groq python client README.
"""

import os
import json
import time
from typing import List
import html  # add at top if not already present

import streamlit as st
from streamlit_lottie import st_lottie
from groq import Client
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ========== Helper / UI functions ==========
def load_lottie_url(url: str):
    """Fetch a Lottie JSON from a URL. streamlit_lottie requires JSON or dict."""
    import requests
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

def local_css(css: str):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Default accessible Groq sandbox/test model
CHAT_MODELS = [
    "groq/compound-mini"
]


# Small CSS to modernize appearance
BASE_CSS = """
:root{
  --accent: #7c3aed;
  --bg: #0f172a;
  --card: #0b1220;
  --muted: #93c5fd;
}
.css-1d391kg { padding: 0.5rem; }
body {
  background: linear-gradient(180deg, #0b1220 0%, #061026 100%);
}
.stButton>button {
  border-radius: 12px;
  background: linear-gradient(90deg, rgba(124,58,237,1), rgba(59,130,246,1));
  color: white;
  padding: 8px 14px;
  font-weight: 600;
}
.streamlit-expanderHeader {
  color: var(--muted);
}
.chat-bubble {
  border-radius: 12px;
  padding: 10px 12px;
  margin: 6px 0;
  max-width: 80%;
}
.user {
  background: linear-gradient(90deg,#34d399,#10b981);
  color: #ffffff;
  margin-left: auto;
  padding: 12px 16px;
  border-radius: 16px;
  max-width: 75%;
  line-height: 1.5;
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}

.bot {
  background: linear-gradient(90deg, #3b82f6, #60a5fa); /* blue gradient */
  color: #ffffff;                                       /* bright white text */
  margin-right: auto;
  padding: 12px 16px;
  border-radius: 16px;
  max-width: 75%;
  line-height: 1.5;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}



.small-muted { color: rgba(255,255,255,0.45); font-size:12px; }
"""

# ========== Groq client wrapper ==========
class GroqWrapper:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("No GROQ API key provided")
        # ✅ Create client (new SDK)
        self.client = Client(api_key=api_key)

    def list_models(self):
        """List available models."""
        try:
            response = self.client.models.list()
            return [m.id for m in response.data]
        except Exception as e:
            return {"error": str(e)}

    def chat(self, model: str, messages: List[dict], max_output_tokens: int = 512):
        """
        Chat endpoint — now uses Groq.chat.completions.create()
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_output_tokens
            )
            # ✅ Extract text content safely
            return response.choices[0].message.content, response
        except Exception as e:
            return f"[ERROR] {e}", {"error": str(e)}

    def embeddings(self, model: str, input_text: str):
        """
        Embeddings endpoint — Groq.embeddings.create()
        """
        try:
            response = self.client.embeddings.create(
                model=model,
                input=input_text
            )
            emb = response.data[0].embedding
            return emb, response
        except Exception as e:
            return None, {"error": str(e)}

# ========== Streamlit app ==========
st.set_page_config(page_title="Groq Playground by Bhavin — Streamlit", layout="wide", initial_sidebar_state="expanded")

local_css(BASE_CSS)

# Top row with a Lottie animation and title
col1, col2 = st.columns([1, 3])
with col1:
    lottie_url = "https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json"
    lottie_json = load_lottie_url(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=200, key="top-lottie")
    else:
        st.image("https://www.gstatic.com/devrel-devsite/prod/v3f0b0d33bdc3a2b9a2b6a0a6a43b5e64b3f1a7e5c7a3b2c1d4e5f6/static/images/og-image.png", width=160)

with col2:
    st.title("Groq Playground by Bhavin")
    st.markdown("A single-file Streamlit demo that calls the Groq API (chat + embeddings). Modern UI with Lottie animation. Paste your API key in the sidebar or set `GROQ_API_KEY` in env.")

# Sidebar inputs
st.sidebar.header("Configuration")
api_key_in = st.sidebar.text_input("Groq API Key (or leave blank to use env var)", type="password")
api_key = api_key_in.strip() or os.getenv("GROQ_API_KEY", "").strip()
api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else api_key

if not api_key:
    st.sidebar.warning("No GROQ_API_KEY detected. Paste your key here or set environment variable GROQ_API_KEY.")
model_choice = st.sidebar.selectbox("Model (suggested)", CHAT_MODELS, index=0)
max_tokens = st.sidebar.slider("Max output tokens", 64, 2048, 512, 64)

# Create client if key is present
client_wrapper = None
if api_key:
    try:
        client_wrapper = GroqWrapper(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Groq client: {e}")

# Show model list button (calls Groq /models endpoint)
with st.expander("Model Explorer"):
    if client_wrapper:
        if st.button("Fetch available models from Groq"):
            with st.spinner("Fetching models..."):
                models = client_wrapper.list_models()
                st.json(models)
    else:
        st.write("Provide an API key to fetch models.")

# Chat area
st.subheader("Chat with the model")
chat_col1, chat_col2 = st.columns([3, 1])

with chat_col1:
    # chat history stored in session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (role, text)
    # render history
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='chat-bubble user'>{html.escape(text)}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble bot'><strong>Model</strong><br/>{text}</div>", unsafe_allow_html=True)

    user_input = st.text_area("Your message", value="", key="user_input", height=120)
    if st.button("Send"):
        if not client_wrapper:
            st.error("No Groq client available — provide API key first.")
        elif not user_input.strip():
            st.warning("Write a message first.")
        else:
            # append user message immediately
            st.session_state.chat_history.append(("user", user_input.strip()))
            # prepare messages structure
            # include a system message optionally
            system_msg = st.sidebar.text_area("System message (optional)", value="You are a helpful assistant.", height=80)
            messages = []
            if system_msg and system_msg.strip():
                messages.append({"role": "system", "content": system_msg.strip()})
            # include last N messages as user/assistant turns (very simple)
            for role, msg in st.session_state.chat_history[-10:]:
                if role == "user":
                    messages.append({"role": "user", "content": msg})
                else:
                    messages.append({"role": "assistant", "content": msg})

            with st.spinner("Calling Groq model..."):
                out_text, raw = client_wrapper.chat(model_choice, messages, max_output_tokens=max_tokens)
            # append model reply
            st.session_state.chat_history.append(("assistant", out_text))
            st.rerun()


with chat_col2:
    st.markdown("### Tools")
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Quick prompts")
    if st.button("Polish text"):
        st.session_state.chat_history.append(("user", "Polish this sentence and make it grammatically correct."))
        st.rerun()
    if st.button("Explain like I'm 5"):
        st.session_state.chat_history.append(("user", "Explain the concept in simple terms as if to a 5-year-old."))
        st.rerun()

    if st.button("Summarize last message"):
        last_user = None
        for role, m in reversed(st.session_state.chat_history):
            if role == "user":
                last_user = m
                break
        if last_user:
            st.session_state.chat_history.append(("user", f"Summarize the following message: {last_user}"))
            st.rerun()

        else:
            st.info("No user message to summarize.")

st.markdown("---")

# Embeddings demo
st.subheader("Embeddings generator")
emb_col1, emb_col2 = st.columns([3, 1])
with emb_col1:
    emb_text = st.text_area("Text to embed", "Groq is lightning fast!", key="emb_text")
    emb_model = st.selectbox("Embedding model (example)", ["text-embedding-3-small", "text-embedding-3-large"], index=0)
    if st.button("Get embedding"):
        if not client_wrapper:
            st.error("No Groq client available.")
        else:
            with st.spinner("Requesting embeddings..."):
                vec, raw = client_wrapper.embeddings(emb_model, emb_text)
            if vec is None:
                st.error(f"Embedding failed: {raw}")
            else:
                st.success(f"Embedding vector length: {len(vec)}")
                st.write("First 10 dims:", vec[:10])
                st.caption("Full embedding shown as JSON below.")
                st.json(raw)

with emb_col2:
    st.markdown("Tip: use embeddings for semantic search, clustering, or RAG pipelines.")

st.markdown("---")

# Simple file upload (basic RAG example)
st.subheader("Upload text files (optional) — include file contents as context")
uploaded = st.file_uploader("Upload .txt files (multiple allowed)", accept_multiple_files=True, type=["txt"])
if uploaded:
    docs = []
    for f in uploaded:
        content = f.read().decode("utf-8", errors="replace")
        docs.append({"name": f.name, "content": content})
    st.write(f"Loaded {len(docs)} document(s). You can include snippets from these files in your chat prompts.")
    # show simple search
    q = st.text_input("Search text across uploaded docs (simple substring search)")
    if q:
        results = []
        for d in docs:
            if q.lower() in d["content"].lower():
                snippet_idx = d["content"].lower().index(q.lower())
                snippet = d["content"][max(0, snippet_idx-80): snippet_idx+len(q)+80]
                results.append((d["name"], snippet))
        if results:
            for name, snip in results:
                st.markdown(f"**{name}** — ...{snip}...")
        else:
            st.info("No matches found.")

st.markdown("---")
st.caption("This is a demo. Replace model strings with the model IDs in your Groq account. If you need streaming/advanced features, adapt the client calls accordingly.")

# Footer / credits and docs links
colf1, colf2 = st.columns([2,4])
with colf1:
    st.write("Made with ❤️ — Streamlit + Groq by Bhavin")
with colf2:
    st.markdown(
        """
        **Helpful links**  
        - Groq quickstart & docs: https://console.groq.com/docs  
        - Groq Python client (install): `pip install groq`  
        """)
