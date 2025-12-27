import streamlit as st
from rag_pipeline import answer_query

st.set_page_config(
    page_title="InsightWeave",
    layout="wide"
)

st.title("🧠 InsightWeave")
st.subheader("Hybrid Knowledge Search & Real-Time RAG Engine")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")

use_web = st.sidebar.toggle("Enable Web Search", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Route Indicators**")
st.sidebar.markdown("📄 Document-based")
st.sidebar.markdown("🌐 Web-based")
st.sidebar.markdown("🔀 Hybrid")

# ---------- Main Chat ----------
query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        result = answer_query(query, use_web=use_web)

    route = result["route"]
    answer = result["answer"]

    if route == "doc":
        icon = "📄"
    elif route == "web":
        icon = "🌐"
    else:
        icon = "🔀"

    st.markdown(f"### {icon} Answer")
    with st.expander("🧾 Final Answer (Summarized)"):
        st.write(answer)

    with st.expander("📚 Retrieved Evidence (Raw Text)"):
        st.write("The following text snippets were retrieved and used to ground the answer:")
        st.code(answer)


