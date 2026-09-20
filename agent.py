"""Layoffs Research Agent — a LangGraph state machine built on top of Week 2's RAG corpus.

Control flow:

    retrieve_corpus --sufficient--> answer_from_corpus --> END
          |
      insufficient
          v
      web_search --no hits--> handle_error --> END
          |
       hits found
          v
   scrape_article --all failed--> handle_error --> END
          |
      scraped ok
          v
  extract_and_draft
          |
          v
  human_approval (interrupt: approve/reject adding this article to the corpus)
          |
     +----+----+
     |         |
  approved   rejected
     |         |
save_to_corpus |
     v         v
        END

Reads (retrieve_corpus, web_search, scrape_article) run autonomously.
The only write — permanently growing the corpus — always waits for a human.
"""
import os
from datetime import date
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from openai import OpenAI

import retrieval
import tools

CHAT_MODEL = os.environ.get("NEBIUS_CHAT_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")

client = OpenAI(
    api_key=os.environ["NEBIUS_API_KEY"],
    base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
)


class AgentState(TypedDict, total=False):
    query: str
    index: dict  # in-memory corpus index (embeddings + meta), passed through, never serialized in checkpoints
    corpus_results: list[dict]
    corpus_score: float
    search_hits: list[dict]
    scraped_url: str | None
    scraped_text: str | None
    candidate_doc: dict | None
    answer: str
    sources: list[dict]
    refused: bool
    approved: bool | None


def retrieve_corpus_node(state: AgentState) -> dict:
    results, score = retrieval.retrieve(state["query"], state["index"])
    return {"corpus_results": results, "corpus_score": score}


def route_after_retrieve(state: AgentState) -> str:
    return "answer_from_corpus" if state["corpus_score"] >= retrieval.MIN_COMBINED_SCORE else "web_search"


def answer_from_corpus_node(state: AgentState) -> dict:
    results = state["corpus_results"]
    context = "\n\n".join(
        f"[{i+1}] {r['company']} ({r['date']}, {r['industry']}): {r['text']}"
        for i, r in enumerate(results)
    )
    prompt = f"""Answer the question using ONLY the numbered sources below. Cite sources inline like [1], [2].
If the sources don't contain the answer, say you don't know — do not use outside knowledge.

Sources:
{context}

Question: {state['query']}

Answer:"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": results,
        "refused": False,
    }


def web_search_node(state: AgentState) -> dict:
    hits = tools.web_search(state["query"])
    return {"search_hits": hits}


def route_after_search(state: AgentState) -> str:
    return "scrape_article" if state["search_hits"] else "handle_error"


def scrape_article_node(state: AgentState) -> dict:
    for hit in state["search_hits"]:
        text = tools.scrape_article(hit["url"])
        if text:
            return {"scraped_url": hit["url"], "scraped_text": text}
    return {"scraped_url": None, "scraped_text": None}


def route_after_scrape(state: AgentState) -> str:
    return "extract_and_draft" if state.get("scraped_text") else "handle_error"


def extract_and_draft_node(state: AgentState) -> dict:
    result = tools.extract_facts_and_answer(state["query"], state["scraped_text"], state["scraped_url"])
    facts = result["facts"]
    existing_count = len(retrieval.load_articles())
    candidate_doc = {
        "id": f"doc_{existing_count:03d}",
        "company": facts.get("company", "unknown"),
        "industry": facts.get("industry", "unknown"),
        "country": "unknown",
        "stage": "unknown",
        "total_laid_off": facts.get("total_laid_off", "unknown"),
        "date": facts.get("date", str(date.today())),
        "url": state["scraped_url"],
        "text": state["scraped_text"],
    }
    return {
        "answer": result["answer"],
        "sources": [{"company": candidate_doc["company"], "date": candidate_doc["date"], "url": state["scraped_url"], "text": state["scraped_text"][:200]}],
        "candidate_doc": candidate_doc,
        "refused": False,
    }


def human_approval_node(state: AgentState) -> dict:
    decision = interrupt(
        {
            "question": "Add this newly researched article to the permanent corpus?",
            "candidate_doc": {k: v for k, v in state["candidate_doc"].items() if k != "text"} | {"text_preview": state["candidate_doc"]["text"][:300]},
        }
    )
    return {"approved": bool(decision.get("approve", False))}


def route_after_approval(state: AgentState) -> str:
    return "save_to_corpus" if state.get("approved") else END


def save_to_corpus_node(state: AgentState) -> dict:
    updated_index = retrieval.add_article_to_index(state["candidate_doc"], state["index"])
    return {"index": updated_index}


def handle_error_node(state: AgentState) -> dict:
    return {
        "answer": (
            "I couldn't find this in the indexed corpus, and my web search/scrape attempts "
            "didn't turn up a readable article either. Try rephrasing, naming the company more "
            "precisely, or asking about a different layoff event."
        ),
        "sources": [],
        "refused": True,
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_corpus", retrieve_corpus_node)
    graph.add_node("answer_from_corpus", answer_from_corpus_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("scrape_article", scrape_article_node)
    graph.add_node("extract_and_draft", extract_and_draft_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("save_to_corpus", save_to_corpus_node)
    graph.add_node("handle_error", handle_error_node)

    graph.add_edge(START, "retrieve_corpus")
    graph.add_conditional_edges(
        "retrieve_corpus", route_after_retrieve, {"answer_from_corpus": "answer_from_corpus", "web_search": "web_search"}
    )
    graph.add_edge("answer_from_corpus", END)
    graph.add_conditional_edges(
        "web_search", route_after_search, {"scrape_article": "scrape_article", "handle_error": "handle_error"}
    )
    graph.add_conditional_edges(
        "scrape_article", route_after_scrape, {"extract_and_draft": "extract_and_draft", "handle_error": "handle_error"}
    )
    graph.add_edge("extract_and_draft", "human_approval")
    graph.add_conditional_edges(
        "human_approval", route_after_approval, {"save_to_corpus": "save_to_corpus", END: END}
    )
    graph.add_edge("save_to_corpus", END)
    graph.add_edge("handle_error", END)

    return graph.compile(checkpointer=InMemorySaver())
