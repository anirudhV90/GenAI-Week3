import uuid

import streamlit as st
from langgraph.types import Command

from agent import build_graph
from retrieval import build_index

st.set_page_config(page_title="Layoffs Research Agent", page_icon="🕵️", layout="wide")

st.title("🕵️ Layoffs Research Agent")
st.caption(
    "A LangGraph agent built on top of the Week 2 RAG corpus. It checks the indexed "
    "layoffs-news articles first; if coverage is thin, it autonomously searches the web, "
    "scrapes an article, and drafts a cited answer — then asks you before permanently "
    "adding that article to the corpus."
)


@st.cache_resource
def get_graph():
    return build_graph()


if "index" not in st.session_state:
    with st.spinner("Loading corpus index..."):
        st.session_state.index = build_index()
if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None

graph = get_graph()
index = st.session_state.index

st.sidebar.header("Corpus")
st.sidebar.markdown(f"**{len(index['meta'])}** chunks indexed")
companies = sorted({m["company"] for m in index["meta"]})
st.sidebar.markdown(f"**{len(companies)}** companies covered")
with st.sidebar.expander("Companies in corpus"):
    st.write(", ".join(companies))
st.sidebar.caption(
    "Grows over time: every web-researched answer you approve below gets embedded "
    "and added here for future questions."
)

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("note"):
            st.caption(turn["note"])
        if turn["sources"]:
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for i, s in enumerate(turn["sources"]):
                    st.markdown(f"**[{i+1}] {s['company']}** — {s['date']} — [{s['url']}]({s['url']})")
                    st.caption(s["text"][:200] + "...")

if st.session_state.pending:
    pending = st.session_state.pending
    doc = pending["value"]["candidate_doc"]
    with st.chat_message("assistant"):
        st.write(pending["draft_answer"])
        st.warning(
            f"**Human approval needed** — add this newly researched article to the permanent corpus?\n\n"
            f"**Company:** {doc['company']} · **Date:** {doc['date']} · **Layoffs:** {doc['total_laid_off']}\n\n"
            f"**URL:** {doc['url']}\n\n*Preview:* {doc['text_preview']}..."
        )
        col1, col2 = st.columns(2)
        approve = col1.button("Approve — add to corpus", type="primary", use_container_width=True)
        reject = col2.button("Reject — answer only, don't save", use_container_width=True)

        if approve or reject:
            result = graph.invoke(Command(resume={"approve": approve}), pending["config"])
            if approve and "index" in result:
                st.session_state.index = result["index"]
            st.session_state.history.append(
                {
                    "question": pending["question"],
                    "answer": result["answer"],
                    "sources": result.get("sources", []),
                    "note": "Added to corpus." if approve else "Not saved to corpus (rejected).",
                }
            )
            st.session_state.pending = None
            st.rerun()

query = st.chat_input(
    "Ask about a company, industry, or layoff event...",
    disabled=bool(st.session_state.pending),
)
if query:
    with st.chat_message("user"):
        st.write(query)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    with st.spinner("Researching..."):
        result = graph.invoke({"query": query, "index": index}, config)

    if "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        st.session_state.pending = {
            "thread_id": thread_id,
            "config": config,
            "value": interrupt_value,
            "question": query,
            "draft_answer": result["answer"],
        }
        st.rerun()
    else:
        st.session_state.history.append(
            {
                "question": query,
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "note": None,
            }
        )
        st.rerun()
