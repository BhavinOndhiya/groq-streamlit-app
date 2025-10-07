"""
groq_streamlit_app.py
Single-file Streamlit app with a modern UI + animations that calls the Groq API.
Features:
 - Paste or read GROQ_API_KEY from env
 - Chat-style interface for generation (choose model)
 - Upload images for vision models
 - Lottie animations and CSS for a modern look
Requirements:
 pip install streamlit groq streamlit-lottie python-dotenv
Docs: https://console.groq.com/docs
"""

import os
import base64
import html
from typing import List

import streamlit as st
from streamlit_lottie import st_lottie
from groq import Groq
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# ========== Helper / UI functions ==========
def load_lottie_url(url: str):
    """Fetch a Lottie JSON from a URL."""
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

# Available Groq models organized by category - UPDATED from API list
REASONING_MODELS = {
    "deepseek-r1-distill-llama-70b": "🧠 DeepSeek R1 Distill Llama 70B - Advanced reasoning (131K context)",
    "openai/gpt-oss-120b": "🧠 GPT OSS 120B - Advanced reasoning (131K context)",
    "openai/gpt-oss-20b": "🧠 GPT OSS 20B - Fast reasoning (131K context)",
    "qwen/qwen3-32b": "🧠 Qwen 3 32B - Multilingual reasoning (131K context)"
}

TEXT_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct": "💬 Llama 4 Scout - Latest Meta model (131K context)",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "💬 Llama 4 Maverick - Advanced text generation (131K context)",
    "llama-3.3-70b-versatile": "💬 Llama 3.3 70B - Versatile text generation (131K context)",
    "llama-3.1-8b-instant": "💬 Llama 3.1 8B Instant - Fast & efficient (131K context)",
    "moonshotai/kimi-k2-instruct": "💬 Kimi K2 Instruct - Conversational AI (131K context)",
    "moonshotai/kimi-k2-instruct-0905": "💬 Kimi K2 Instruct 0905 - Latest version (262K context)",
    "allam-2-7b": "💬 Allam 2 7B - Compact model (4K context)",
    "gemma2-9b-it": "💬 Gemma 2 9B IT - Google's efficient model (8K context)"
}

VISION_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct": "📸 Llama 4 Scout - Vision + Text (131K context)",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "📸 Llama 4 Maverick - Advanced vision (131K context)"
}

FUNCTION_CALLING_MODELS = {
    "openai/gpt-oss-120b": "🔧 GPT OSS 120B - Function calling (131K context)",
    "openai/gpt-oss-20b": "🔧 GPT OSS 20B - Tool use (131K context)",
    "meta-llama/llama-4-scout-17b-16e-instruct": "🔧 Llama 4 Scout - Function calling (131K context)",
    "qwen/qwen3-32b": "🔧 Qwen 3 32B - Tool use (131K context)"
}

MULTILINGUAL_MODELS = {
    "openai/gpt-oss-120b": "🌍 GPT OSS 120B - Multilingual (131K context)",
    "openai/gpt-oss-20b": "🌍 GPT OSS 20B - Multilingual (131K context)",
    "moonshotai/kimi-k2-instruct": "🌍 Kimi K2 - Multilingual (131K context)",
    "moonshotai/kimi-k2-instruct-0905": "🌍 Kimi K2 0905 - Multilingual (262K context)",
    "meta-llama/llama-4-scout-17b-16e-instruct": "🌍 Llama 4 Scout - Multilingual (131K context)",
    "llama-3.3-70b-versatile": "🌍 Llama 3.3 70B - Multilingual (131K context)",
    "qwen/qwen3-32b": "🌍 Qwen 3 32B - Multilingual (131K context)",
    "allam-2-7b": "🌍 Allam 2 7B - Arabic focus (4K context)"
}

AUDIO_MODELS = {
    "whisper-large-v3": "🎤 Whisper Large v3 - Speech to Text",
    "whisper-large-v3-turbo": "🎤 Whisper Large v3 Turbo - Fast speech to text",
    "playai-tts": "🔊 PlayAI TTS - Text to Speech (English)",
    "playai-tts-arabic": "🔊 PlayAI TTS Arabic - Text to Speech (Arabic)"
}

SAFETY_MODELS = {
    "meta-llama/llama-guard-4-12b": "🛡️ Llama Guard 4 12B - Content moderation (131K context)",
    "meta-llama/llama-prompt-guard-2-86m": "🛡️ Llama Prompt Guard 2 86M - Prompt injection detection",
    "meta-llama/llama-prompt-guard-2-22m": "🛡️ Llama Prompt Guard 2 22M - Lightweight prompt guard"
}

COMPOUND_MODELS = {
    "groq/compound": "⚡ Groq Compound - Multi-model routing (131K context)",
    "groq/compound-mini": "⚡ Groq Compound Mini - Fast routing (131K context)"
}

# Model descriptions for help text
MODEL_DESCRIPTIONS = {
    "openai/gpt-oss-120b": "120B parameter model with 131K context window. Advanced reasoning, function calling, and multilingual capabilities. Max output: 65K tokens.",
    "openai/gpt-oss-20b": "20B parameter model with 131K context window. Optimized for speed while maintaining quality. Max output: 65K tokens.",
    "qwen/qwen3-32b": "32B multilingual model with 131K context window. Excellent for reasoning and function calling across languages. Max output: 40K tokens.",
    "moonshotai/kimi-k2-instruct": "Advanced conversational AI with 131K context window. Strong multilingual support. Max output: 16K tokens.",
    "moonshotai/kimi-k2-instruct-0905": "Latest Kimi K2 with 262K context window (largest available). Max output: 16K tokens.",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Latest Llama 4 Scout with 131K context. Supports text, vision, and function calling. Max output: 8K tokens.",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Advanced Llama 4 Maverick with 131K context. Superior image understanding. Max output: 8K tokens.",
    "llama-3.3-70b-versatile": "Versatile 70B parameter model with 131K context. Reliable for various text tasks. Max output: 32K tokens.",
    "llama-3.1-8b-instant": "Fast 8B model with 131K context. Instant responses with good quality. Max output: 131K tokens.",
    "deepseek-r1-distill-llama-70b": "70B reasoning model with 131K context. Distilled from DeepSeek R1. Max output: 131K tokens.",
    "allam-2-7b": "7B model by SDAIA with 4K context. Compact and efficient. Max output: 4K tokens.",
    "gemma2-9b-it": "Google's 9B instruction-tuned model with 8K context. Max output: 8K tokens.",
    "whisper-large-v3": "OpenAI's Whisper large model for high-quality speech-to-text transcription.",
    "whisper-large-v3-turbo": "Faster variant of Whisper large with optimized performance for real-time use.",
    "playai-tts": "High-quality English text-to-speech synthesis from PlayAI.",
    "playai-tts-arabic": "High-quality Arabic text-to-speech synthesis from PlayAI.",
    "meta-llama/llama-guard-4-12b": "12B safety model with 131K context for content moderation and harmful content detection.",
    "meta-llama/llama-prompt-guard-2-86m": "86M parameter model for detecting prompt injection attacks.",
    "meta-llama/llama-prompt-guard-2-22m": "22M parameter lightweight model for prompt security.",
    "groq/compound": "Intelligent routing system that automatically selects the best model for your query.",
    "groq/compound-mini": "Faster variant of Compound with optimized routing for speed."
}

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
  background: linear-gradient(90deg, #3b82f6, #60a5fa);
  color: #ffffff;
  margin-right: auto;
  padding: 12px 16px;
  border-radius: 16px;
  max-width: 75%;
  line-height: 1.5;
  box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}

.small-muted { color: rgba(255,255,255,0.45); font-size:12px; }
.model-category {
  background: rgba(124,58,237,0.1);
  border-left: 3px solid #7c3aed;
  padding: 8px 12px;
  margin: 8px 0;
  border-radius: 6px;
}
"""

# ========== Groq client wrapper ==========
class GroqWrapper:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("No GROQ API key provided")
        self.client = Groq(api_key=api_key)

    def chat(self, model: str, messages: List[dict], max_output_tokens: int = 512, image_data: bytes = None):
        """
        Chat endpoint with optional image input for vision models
        """
        try:
            # If image is provided, convert to base64 and add to the last user message
            if image_data:
                base64_image = base64.b64encode(image_data).decode('utf-8')
                # Find the last user message and make it multimodal
                for msg in reversed(messages):
                    if msg["role"] == "user":
                        # Convert text content to multimodal format
                        text_content = msg["content"]
                        msg["content"] = [
                            {
                                "type": "text",
                                "text": text_content
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                        break

            # Call the chat completions endpoint
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=0.7
            )

            # Extract text from response
            text_out = response.choices[0].message.content
            return text_out, response

        except Exception as e:
            return f"[ERROR] {e}", {"error": str(e)}

    def list_models(self):
        """List available models"""
        try:
            models = self.client.models.list()
            return [{"id": m.id, "owned_by": m.owned_by} for m in models.data]
        except Exception as e:
            return {"error": str(e)}

# ========== Streamlit app ==========
st.set_page_config(
    page_title="Groq Playground by Bhavin — Streamlit", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

local_css(BASE_CSS)

# Top row with a Lottie animation and title
col1, col2 = st.columns([1, 3])
with col1:
    lottie_url = "https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json"
    lottie_json = load_lottie_url(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=200, key="top-lottie")

with col2:
    st.title("🚀 Groq Playground by Bhavin")
    st.markdown("A single-file Streamlit demo that calls the Groq API. Modern UI with Lottie animation. Paste your API key in the sidebar or set `GROQ_API_KEY` in env.")

# Sidebar inputs
st.sidebar.header("⚙️ Configuration")
api_key_in = st.sidebar.text_input("Groq API Key (or leave blank to use env var)", type="password")
api_key = api_key_in.strip() or os.getenv("GROQ_API_KEY", "").strip()

# Check for Streamlit secrets
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    st.sidebar.warning("⚠️ No GROQ_API_KEY detected. Paste your key here or set environment variable GROQ_API_KEY.")

# Model selection by category
st.sidebar.subheader("🎯 Select Model Category")
model_category = st.sidebar.radio(
    "Choose category:",
    ["💬 Text Generation", "📸 Vision", "🧠 Reasoning", "🔧 Function Calling", "🌍 Multilingual", "🎤 Audio", "🛡️ Safety", "⚡ Compound (Routing)"],
    label_visibility="collapsed"
)

# Get models based on category
if model_category == "💬 Text Generation":
    available_models = TEXT_MODELS
elif model_category == "📸 Vision":
    available_models = VISION_MODELS
elif model_category == "🧠 Reasoning":
    available_models = REASONING_MODELS
elif model_category == "🔧 Function Calling":
    available_models = FUNCTION_CALLING_MODELS
elif model_category == "🌍 Multilingual":
    available_models = MULTILINGUAL_MODELS
elif model_category == "🎤 Audio":
    available_models = AUDIO_MODELS
elif model_category == "⚡ Compound (Routing)":
    available_models = COMPOUND_MODELS
else:  # Safety
    available_models = SAFETY_MODELS

model_choice = st.sidebar.selectbox(
    "Select Model",
    options=list(available_models.keys()),
    format_func=lambda x: available_models[x],
    index=0
)

# Show model description
if model_choice in MODEL_DESCRIPTIONS:
    st.sidebar.info(f"ℹ️ {MODEL_DESCRIPTIONS[model_choice]}")

max_tokens = st.sidebar.slider("Max output tokens", 64, 2048, 512, 64)

# System message
system_msg = st.sidebar.text_area(
    "System message (optional)", 
    value="You are a helpful assistant.", 
    height=80
)

# Create client if key is present
client_wrapper = None
if api_key:
    try:
        client_wrapper = GroqWrapper(api_key=api_key)
        st.sidebar.success("✅ Client initialized")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to initialize Groq client: {e}")

# Show model list button
with st.expander("🔍 Model Explorer - View All Available Models"):
    st.markdown("### 📋 All Available Models by Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💬 Text Generation")
        for model_id, desc in TEXT_MODELS.items():
            st.markdown(f"- `{model_id}`")
            st.caption(desc)
        
        st.markdown("#### 🧠 Reasoning")
        for model_id, desc in REASONING_MODELS.items():
            st.markdown(f"- `{model_id}`")
            st.caption(desc)
        
        st.markdown("#### 🔧 Function Calling")
        for model_id, desc in FUNCTION_CALLING_MODELS.items():
            st.markdown(f"- `{model_id}`")
            st.caption(desc)
        
        st.markdown("#### ⚡ Compound (Routing)")
        for model_id, desc in COMPOUND_MODELS.items():
            st.markdown(f"- `{model_id}`")
            st.caption(desc)
    
    with col2:
        st.markdown("#### 📸 Vision")
        for model_id, desc in VISION_MODELS.items():
            st.markdown(f"- `{model_id}`")
            st.caption(desc)
        
        st.markdown("#### 🌍 Multilingual")
        for model_id, desc in MULTILINGUAL_MODELS.items():
            st.markdown(f"- `{model_id}`")
            st.caption(desc)
        
        st.markdown("#### 🎤 Audio")
        for model_id, desc in AUDIO_MODELS.items():
            st.markdown(f"- `{model_id}`")
            st.caption(desc)
        
        st.markdown("#### 🛡️ Safety")
        for model_id, desc in SAFETY_MODELS.items():
            st.markdown(f"- `{model_id}`")
            st.caption(desc)
    
    st.markdown("---")
    if client_wrapper:
        if st.button("🔄 Fetch live models from Groq API"):
            with st.spinner("Fetching models..."):
                models = client_wrapper.list_models()
                st.json(models)
    else:
        st.write("Provide an API key to fetch live model list from API.")

# Chat area
st.subheader("💬 Chat with the model")
chat_col1, chat_col2 = st.columns([3, 1])

with chat_col1:
    # Chat history stored in session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render history
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='chat-bubble user'>{html.escape(text)}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble bot'><strong>🤖 Model</strong><br/>{text}</div>", unsafe_allow_html=True)

    # Chat input
    user_input = st.text_area("Your message", value="", key="user_input", height=120)

    # Image upload (only show for vision models)
    image_file = None
    is_vision_model = model_choice in VISION_MODELS
    
    if is_vision_model:
        st.info("📸 This model supports vision! You can upload images along with your text.")
        uploaded_image = st.file_uploader(
            "Upload an image (PNG, JPG, JPEG)", 
            type=["png", "jpg", "jpeg"],
            help="Upload an image for the vision model to analyze"
        )
        if uploaded_image:
            image_file = uploaded_image.read()
            st.image(image_file, caption="Uploaded Image", use_container_width=True)
    else:
        st.info(f"💬 Currently using: **{model_choice}** - This is a text-only model. Switch to Vision category (📸) to upload images.")

    if st.button("📤 Send", type="primary"):
        if not client_wrapper:
            st.error("❌ No Groq client available — provide API key first.")
        elif not user_input.strip() and not image_file:
            st.warning("⚠️ Write a message or upload an image first.")
        else:
            # Append user message to chat history
            if user_input.strip():
                st.session_state.chat_history.append(("user", user_input.strip()))
            elif image_file:
                st.session_state.chat_history.append(("user", "[Image uploaded]"))

            # Prepare messages
            messages = []
            if system_msg and system_msg.strip():
                messages.append({"role": "system", "content": system_msg.strip()})
            
            for role, msg in st.session_state.chat_history[-10:]:
                if role == "user":
                    messages.append({"role": "user", "content": msg})
                else:
                    messages.append({"role": "assistant", "content": msg})

            # Call Groq model with optional image
            with st.spinner("🔄 Calling Groq model..."):
                out_text, raw = client_wrapper.chat(
                    model_choice, 
                    messages, 
                    max_output_tokens=max_tokens, 
                    image_data=image_file if is_vision_model else None
                )

            st.session_state.chat_history.append(("assistant", out_text))
            st.rerun()

with chat_col2:
    st.markdown("### 🛠️ Tools")
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚡ Quick prompts")
    if st.button("✨ Polish text"):
        st.session_state.chat_history.append(("user", "Polish this sentence and make it grammatically correct."))
        st.rerun()
    
    if st.button("🧒 Explain like I'm 5"):
        st.session_state.chat_history.append(("user", "Explain the concept in simple terms as if to a 5-year-old."))
        st.rerun()

    if st.button("📝 Summarize last message"):
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

# Simple file upload (basic RAG example)
st.subheader("📁 Upload text files (optional) — include file contents as context")
uploaded = st.file_uploader("Upload .txt files (multiple allowed)", accept_multiple_files=True, type=["txt"])
if uploaded:
    docs = []
    for f in uploaded:
        content = f.read().decode("utf-8", errors="replace")
        docs.append({"name": f.name, "content": content})
    st.write(f"✅ Loaded {len(docs)} document(s). You can include snippets from these files in your chat prompts.")
    
    # Show simple search
    q = st.text_input("🔎 Search text across uploaded docs (simple substring search)")
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
st.caption("💡 Updated with current Groq models from API. Last updated: October 2025. Context windows range from 4K to 262K tokens.")

# Footer
colf1, colf2 = st.columns([2, 4])
with colf1:
    st.write("Made with ❤️ by Bhavin")
with colf2:
    st.markdown(
        """
        **📚 Helpful links**  
        - [Groq Docs](https://console.groq.com/docs)  
        - [Model Deprecations](https://console.groq.com/docs/deprecations)
        - [Install Groq SDK](https://pypi.org/project/groq/): `pip install groq`  
        """
    )