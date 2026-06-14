import os
import uuid
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Python Q&A Assistant", page_icon="🐍", layout="centered")
st.title("🐍 Python Q&A Assistant")
st.caption("Grounded answers from Stack Overflow · Powered by Groq + LangGraph")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['question_title']}** (score: {src['score']})")

if prompt := st.chat_input("Ask a Python question...", max_chars=2000, disabled=st.session_state.is_loading):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.is_loading = True
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": prompt, "session_id": st.session_state.session_id},
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()

                st.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander("Sources"):
                        for src in data["sources"]:
                            st.markdown(f"- **{src['question_title']}** (score: {src['score']})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                })

            except httpx.HTTPStatusError as e:
                st.error(f"Backend error {e.response.status_code}: {e.response.text}")
            except httpx.RequestError as e:
                st.error(f"Could not reach backend: {e}")
            finally:
                st.session_state.is_loading = False
