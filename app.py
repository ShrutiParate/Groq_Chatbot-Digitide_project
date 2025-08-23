import os
import time
from typing import Iterator, List, Dict
import streamlit as st
from dotenv import load_dotenv

# --- Load API key from .env (local) or secrets (cloud) ---
load_dotenv()

groq_key = None
try:
    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_key:
    groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error("❌ No GROQ_API_KEY found. Please set it in .env (local) or Streamlit secrets.")
    st.stop()

# --- Streamlit page config ---
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="centered")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Chatbot Settings")

MODEL = "llama-3.1-8b-instant"

temperature = st.sidebar.slider("🎲 Temperature", 0.0, 1.5, 0.7, 0.1)
max_tokens = st.sidebar.slider("📏 Max tokens", 256, 4096, 1024, 64)
top_p = st.sidebar.slider("🔎 Top-p (nucleus sampling)", 0.1, 1.0, 0.9, 0.05)

st.sidebar.markdown("---")

# Personas
personas = {
    "🤝 Friendly Assistant": "You are a helpful and friendly assistant. Keep answers simple and warm.",
    "👨‍🏫 Teacher": "You are a teacher who explains concepts clearly with step-by-step examples.",
    "🛠 Tech Support": "You are a tech support agent. Ask clarifying questions and give solutions.",
    "🧘 Philosopher": "You are a deep thinker. Provide thoughtful, reflective answers.",
    "✍️ Custom": ""  # For manual editing
}

persona_choice = st.sidebar.selectbox("🎭 Bot Persona", list(personas.keys()))

# System prompt
system_prompt = personas[persona_choice]
if persona_choice == "✍️ Custom":
    system_prompt = st.sidebar.text_area("✍️ Enter custom system prompt:", height=120)

clear = st.sidebar.button("🧹 Clear chat history")

# --- Session state for chat history ---
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

if clear:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Update system prompt if changed
if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
    if st.session_state.messages[0]["content"] != system_prompt:
        st.session_state.messages[0]["content"] = system_prompt

# --- Header ---
st.title("🤖 Chatbot")
st.caption(f"Provider: **Groq** · Model: **{MODEL}** · Streaming enabled")

# --- Streaming function ---
def stream_groq(messages: List[Dict[str, str]]) -> Iterator[str]:
    from groq import Groq  # pip install groq
    client = Groq(api_key=groq_key)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta

# --- Render chat history ---
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f"🧑 **You:** {msg['content']}")
        else:
            st.markdown(f"🤖 **Bot:** {msg['content']}")

# --- Chat input ---
prompt = st.chat_input("Type your message…")

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(f"🧑 **You:** {prompt}")

    with st.chat_message("assistant"):
        placeholder = st.empty()
        partial = ""
        try:
            generator = stream_groq(st.session_state.messages)
            for token in generator:
                partial += token
                placeholder.markdown(f"🤖 **Bot:** {partial}")
                time.sleep(0.01)
        except Exception as e:
            partial = f"❌ **Error:** {e}"
            placeholder.markdown(partial)

        st.session_state.messages.append({"role": "assistant", "content": partial})
