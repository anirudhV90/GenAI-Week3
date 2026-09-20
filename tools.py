"""Tools the agent reaches for when the corpus doesn't cover a question.

Read-only tools: web_search, scrape_article, extract_facts.
Write tool: none here — the actual corpus write (retrieval.add_article_to_index)
only ever runs after a human approves it in the LangGraph interrupt step.
"""
import os

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from openai import OpenAI

TIMEOUT = 6
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
CHAT_MODEL = os.environ.get("NEBIUS_CHAT_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")

client = OpenAI(
    api_key=os.environ["NEBIUS_API_KEY"],
    base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Read-only. Returns [] on failure or empty results rather than raising —
    the graph treats that as 'nothing found', not a crash."""
    try:
        with DDGS() as ddgs:
            hits = ddgs.text(f"{query} layoffs", max_results=max_results)
        return [{"title": h.get("title", ""), "url": h.get("href", h.get("link", ""))} for h in hits if h.get("href") or h.get("link")]
    except Exception:
        return []


def extract_article_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n".join(p for p in paragraphs if len(p) > 40).strip()


def scrape_article(url: str) -> str | None:
    """Read-only. Returns None on any failure (timeout, non-200, paywall-thin text)
    so the caller can move on to the next search result instead of erroring out."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None
        text = extract_article_text(resp.text)
        return text if len(text) >= 300 else None
    except requests.RequestException:
        return None


def extract_facts_and_answer(query: str, article_text: str, url: str) -> dict:
    """One LLM call that both pulls structured facts (for a future corpus record)
    and drafts a cited answer to the user's question from this single fresh article."""
    prompt = f"""You are researching a tech company layoff for a journalist. You have ONE freshly
scraped news article. Using ONLY this article (no outside knowledge), do two things:

1. Extract these fields as JSON: company, industry (best guess, one or two words), date
   (YYYY-MM-DD if stated, else best guess or "unknown"), total_laid_off (integer or "unknown").
2. Write a short answer (2-4 sentences) to the question below, citing the article as [1].
   If the article does not actually answer the question, say so plainly instead of guessing.

Respond in this exact format, nothing else:
FACTS: {{"company": "...", "industry": "...", "date": "...", "total_laid_off": "..."}}
ANSWER: <your answer text>

Article ({url}):
{article_text[:4000]}

Question: {query}"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""

    facts_raw, answer = {}, content
    if "FACTS:" in content and "ANSWER:" in content:
        facts_part = content.split("FACTS:", 1)[1].split("ANSWER:", 1)[0].strip()
        answer = content.split("ANSWER:", 1)[1].strip()
        import json

        try:
            facts_raw = json.loads(facts_part)
        except json.JSONDecodeError:
            facts_raw = {}

    return {"facts": facts_raw, "answer": answer}
