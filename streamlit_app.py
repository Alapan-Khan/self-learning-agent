import streamlit as st
from agent_core import chat, get_all_memories, clear_all_memories

st.set_page_config(
    page_title="Self-Learning AI Agent",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

USER_ID = "demo_user"

# ---------- Custom styling ----------
st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stAppDeployButton"] {display: none !important;}

    header[data-testid="stHeader"] {
        background-color: #0f0f10 !important;
    }

    header[data-testid="stHeader"] svg {
        fill: #ececec !important;
    }

    header[data-testid="stHeader"] button {
        background-color: transparent !important;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stDecoration"] {
        background-color: #0f0f10 !important;
    }

    [data-testid="stBottom"],
    [data-testid="stBottom"] * ,
    [data-testid="stBottomBlockContainer"],
    .stChatFloatingInputContainer {
        background-image: none !important;
        background-color: #0f0f10 !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        border: none !important;
    }

    [data-testid="stChatInput"] {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem 0 1.5rem 0 !important;
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        border: none !important;
        color: #ececec !important;
        font-size: 1.05rem !important;
    }

    [data-testid="stChatInput"] div {
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] > div:first-child {
        border: 1px solid #3a3b3f !important;
        border-radius: 26px !important;
        background-color: #1e1f22 !important;
        padding: 4px 8px 4px 20px !important;
    }

    [data-testid="stChatInput"] *,
    [data-testid="stChatInput"] *:focus,
    [data-testid="stChatInput"] *:focus-visible,
    [data-testid="stChatInput"] *:focus-within,
    [data-testid="stChatInput"] *:active {
        outline: none !important;
    }

    [data-testid="stChatInput"] label {
        background: transparent !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #ececec !important;
        border-radius: 50% !important;
        width: 34px !important;
        height: 34px !important;
        min-width: 34px !important;
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
    }

    [data-testid="stChatInput"] button svg {
        display: none !important;
    }

    [data-testid="stChatInput"] button::after {
        content: "↑";
        color: #0f0f10 !important;
        font-size: 1.3rem;
        font-weight: 700;
        line-height: 1;
    }

    .block-container {
        max-width: 800px;
        padding-top: 2rem;
    }

    .app-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }
    .app-header h1 {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ececec;
        margin: 0;
    }
    .app-subtitle {
        color: #9a9aa6;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }

    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.5rem 0;
    }

    [data-testid="stChatMessageContent"] {
        border-radius: 14px;
        padding: 14px 18px;
        font-size: 1.08rem;
        line-height: 1.6;
    }

    [data-testid="stChatMessageContent"] p {
        font-size: 1.08rem;
        line-height: 1.6;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
        [data-testid="stChatMessageContent"] {
        background-color: #2a2b32;
        color: #ececec;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
        [data-testid="stChatMessageContent"] {
        background-color: transparent;
        color: #d1d1d6;
        border: 1px solid #2a2b32;
    }

    section[data-testid="stSidebar"] {
        background-color: #171718 !important;
        border-right: 1px solid #2a2b32;
    }
    section[data-testid="stSidebar"] h3 {
        color: #ececec;
        font-size: 1.15rem;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #b4b4bb;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="app-header">
    <h1>🧠 Self-Learning Agent</h1>
</div>
<div class="app-subtitle">Remembers what you tell it — powered by Mem0 + Qdrant + Groq</div>
""", unsafe_allow_html=True)

# ---------- Sidebar: memory panel ----------
with st.sidebar:
    st.subheader("🗂️ Memory")
    memories = get_all_memories(USER_ID)
    if memories:
        for m in memories:
            st.markdown(f"• {m}")
        if st.button("🗑️ Clear Memory"):
            clear_all_memories(USER_ID)
            st.rerun()
    else:
        st.caption("Nothing stored yet — start chatting.")

# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message the agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Thinking..."):
            reply = chat(prompt, USER_ID)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()