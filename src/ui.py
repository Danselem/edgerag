import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("EDGE_RAG_API_URL", "http://localhost:8000")


# ── API calls ─────────────────────────────────────────────────


def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def api_query(query: str, shard_type: str, limit: int) -> dict:
    r = requests.post(
        f"{API_BASE_URL}/api/v1/query",
        json={"query": query, "shard_type": shard_type, "limit": limit},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def api_config() -> dict:
    r = requests.get(f"{API_BASE_URL}/api/v1/config", timeout=5)
    r.raise_for_status()
    return r.json()


# ── Page config ───────────────────────────────────────────────


st.set_page_config(
    page_title="Edge RAG",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────


with st.sidebar:
    st.header("Settings")

    if api_health():
        st.success("API connected")
    else:
        st.error("API not reachable")
        st.caption(f"Expected at {API_BASE_URL}")
        st.stop()

    shard_type = st.selectbox("Index type", ["dense", "hybrid", "quantized"], index=0)
    limit = st.slider("Retrieval limit", 1, 10, 2)
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    try:
        cfg = api_config()
        st.caption(f"Embedding: {cfg['embedding_model']}")
        st.caption(f"LLM: {cfg['llm_model']}")
        st.caption(f"Document: {cfg['document']}")
    except Exception:
        st.caption("Could not load config")

# ── Main area ─────────────────────────────────────────────────


st.title("J.P. Morgan Mid-Year Outlook 2026")
st.caption("Ask questions about the investment outlook. Answers are grounded in the document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "context" in message:
            with st.expander("Retrieved context"):
                st.text(message["context"])

if prompt := st.chat_input("Ask something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = api_query(prompt, shard_type, limit)
                answer = result["answer"]
                context = result["context"][0] if result["context"] else ""

                st.write(answer)
                with st.expander("Retrieved context"):
                    st.text(context)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "context": context}
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {e}"}
                )
