from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_chatbot.chat.service import get_chatbot_service  # noqa: E402


st.set_page_config(page_title="Trợ lý Pháp luật Việt Nam", page_icon="⚖️", layout="wide")


@st.cache_resource
def load_service():
    return get_chatbot_service()


service = load_service()

st.markdown(
    "<h2 style='text-align: center;'>Trợ lý Pháp luật Việt Nam 🇻🇳</h2>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message.get("role", "assistant")
    if role not in {"user", "assistant"}:
        continue
    with st.chat_message(role):
        st.markdown(message.get("content", ""))

user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    chat_history = service.chat(user_input)

    assistant_reply = ""
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            assistant_reply = msg.get("content", "")
            break

    if assistant_reply:
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    st.experimental_rerun()

col1, col2, _ = st.columns([1, 1, 4])

with col1:
    if st.button("🔄 Đoạn chat mới"):
        service.reset()
        st.session_state.messages = []
        st.experimental_rerun()

with col2:
    st.markdown("&nbsp;")

st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>"
    "🚀 Được phát triển bởi nhóm OPENAPI để hỗ trợ pháp luật"
    "</p>",
    unsafe_allow_html=True,
)

