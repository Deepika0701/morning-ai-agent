import json
import os
import re
import smtplib
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from pathlib import Path

import feedparser
import requests
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi

HISTORY_FILE = Path("seen_history.json")
MAX_HISTORY_ITEMS = 60

INTERVIEW_CHANNELS = {
    "Dwarkesh Patel": "UCXl4i9dYBrFOabk0xGmbkRA",
    "Latent Space": "UCcH12M3qp5C9hB2U8U1Z3gQ",
    "No Priors (Sarah Guo & Elad Gil)": "UCn8zNIfYAQNdrFRrr8oibKw",
    "Lex Fridman": "UCSHZKyawb77ixDdsGog4iWA",
    "Y Combinator": "UCcefcZRL2oaA_uBNeo5UOWg",
}

AI_NEWS_FEEDS = [
    "https://hnrss.org/frontpage?q=AI+OR+LLM+OR+agent",
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://huggingface.co/blog/feed.xml",
]

AI_TOOLS_FEEDS = [
    "https://hnrss.org/show?q=AI+OR+LLM+OR+agent+OR+MCP",
    "https://www.producthunt.com/feed?category=artificial-intelligence",
    "https://mshibanami.github.io/GitHubTrendingRSS/daily/python.xml",
    "https://mshibanami.github.io/GitHubTrendingRSS/daily/typescript.xml",
]


def load_history() -> dict:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": [], "tools": [], "interviews": []}


def save_history(history: dict) -> None:
    for key in ("models", "tools", "interviews"):
        history[key] = history.get(key, [])[-MAX_HISTORY_ITEMS:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def fetch_unseen_hf_models(seen_models: list[str], limit: int = 8) -> list[dict]:
    """Fetches recent papers/models from Hugging Face Daily Papers API."""
    candidates = []
    try:
        resp = requests.get("https://huggingface.co/api/daily_papers", timeout=15)
        resp.raise_for_status()
        for item in resp.json():
            paper = item.get("paper", {})
            paper_id = paper.get("id")
            if not paper_id or paper_id in seen_models:
                continue
            candidates.append({
                "id": paper_id,
                "title": item.get("title") or paper.get("title"),
                "summary": (paper.get("summary") or "")[:600],
                "url": f"https://huggingface.co/papers/{paper_id}",
            })
            if len(candidates) >= limit:
                break
    except Exception as exc:
        print(f"[Warn] HF Daily Papers fetch failed: {exc}")
    return candidates


def fetch_rss_items(feed_urls: list[str], seen_urls: list[str], limit: int = 10) -> list[dict]:
    """Fetches fresh unseen items from a list of RSS/Atom feeds."""
    items = []
    for rss_url in feed_urls:
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]:
                link = getattr(entry, "link", "")
                if not link or link in seen_urls:
                    continue
                items.append({
                    "title": getattr(entry, "title", ""),
                    "url": link,
                    "summary": re.sub(r"<[^>]+>", " ", getattr(entry, "summary", ""))[:400].strip(),
                })
                if len(items) >= limit:
                    return items
        except Exception as exc:
            print(f"[Warn] RSS fetch failed for {rss_url}: {exc}")
    return items


def fetch_unseen_interviews(seen_interviews: list[str], limit: int = 3) -> list[dict]:
    """Pulls the latest unseen AI interviews via YouTube RSS + transcript excerpt."""
    interviews = []
    for channel_name, channel_id in INTERVIEW_CHANNELS.items():
        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:2]:
                video_id = getattr(entry, "yt_videoid", None)
                if not video_id or video_id in seen_interviews:
                    continue

                transcript_snippet = "Transcript unavailable; analyze based on title and channel focus."
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                    full_text = " ".join(chunk["text"] for chunk in transcript[40:350])
                    transcript_snippet = full_text[:3500]
                except Exception:
                    pass

                interviews.append({
                    "id": video_id,
                    "channel": channel_name,
                    "title": entry.title,
                    "url": entry.link,
                    "published": getattr(entry, "published", ""),
                    "transcript_excerpt": transcript_snippet,
                })
                if len(interviews) >= limit:
                    return interviews
        except Exception as exc:
            print(f"[Warn] RSS fetch failed for {channel_name}: {exc}")
    return interviews


def parse_json_response(raw_text: str) -> dict:
    """Safely extracts and parses JSON even if wrapped in markdown fences or <think> tags."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def discover_openrouter_free_models(api_key: str, limit: int = 5) -> list[str]:
    """Queries OpenRouter's live catalog for currently active :free text models."""
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        models = resp.json().get("data", [])
        free_ids = []
        for m in models:
            mid = m.get("id", "")
            pricing = m.get("pricing", {})
            is_free = mid.endswith(":free") or (
                str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0"
            )
            if is_free and not any(skip in mid.lower() for skip in ("vision", "embed", "vl-", "guard")):
                free_ids.append(mid)

        # Rank strong open-source families first (Qwen, Llama, DeepSeek, Gemma, Mistral)
        preferred = ("qwen", "llama", "deepseek", "gemma", "mistral", "openrouter/free")
        free_ids.sort(key=lambda x: next((i for i, k in enumerate(preferred) if k in x.lower()), 99))
        return free_ids[:limit]
    except Exception as exc:
        print(f"[Warn] OpenRouter model discovery failed: {exc}")
        return ["openrouter/free"]


def discover_groq_models(api_key: str, limit: int = 4) -> list[str]:
    """Queries Groq's live /models endpoint so retired slugs never cause 404s."""
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        models = [m.get("id", "") for m in resp.json().get("data", [])]
        chat_models = [
            mid for mid in models
            if mid and not any(skip in mid.lower() for skip in ("whisper", "guard", "tts", "embed", "playai"))
        ]
        preferred = ("llama", "qwen", "deepseek", "gemma", "mistral")
        chat_models.sort(key=lambda x: next((i for i, k in enumerate(preferred) if k in x.lower()), 99))
        return chat_models[:limit]
    except Exception as exc:
        print(f"[Warn] Groq model discovery failed: {exc}")
        return []


def call_openai_compatible(
    endpoint_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    extra_headers: dict | None = None,
) -> dict:
    """Calls OpenRouter, Groq, or GitHub Models via standard OpenAI-compatible REST."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Principal AI Systems Architect. "
                    "Respond ONLY with a valid JSON object matching the user's schema. "
                    "Do not include markdown fences or any text outside the JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=90)
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text[:300]}")
    raw_text = resp.json()["choices"][0]["message"]["content"]
    return parse_json_response(raw_text)


def run_digest_agent(
    history: dict,
    hf_candidates: list[dict],
    news_candidates: list[dict],
    tool_candidates: list[dict],
    interview_candidates: list[dict],
) -> dict:
    """Prioritizes Free Open-Source Models (OpenRouter -> Groq -> GitHub Models) with Gemini last."""
    today_str = datetime.now(timezone.utc).strftime("%B %d, %Y")

    prompt = f"""
You are a Principal AI Systems Architect preparing an actionable, engineering-focused morning AI briefing for {today_str}.
Your reader is a software engineer who cares deeply about **practical implementation, system architecture, code/CLI workflows, and real-world use cases**—not marketing hype.

### 1. ALREADY SEEN LEDGER (DO NOT REPEAT THESE IN DEEP DIVES)
{json.dumps(history, indent=2)}

### 2. PRE-FETCHED LIVE CANDIDATES FROM TODAY
- Breaking AI News Feeds (Hacker News, TechCrunch, The Verge, Hugging Face Blog):
{json.dumps(news_candidates, indent=2)}

- New AI Tools & Trending Open-Source Repos (Show HN, Product Hunt AI, GitHub Trending):
{json.dumps(tool_candidates, indent=2)}

- Hugging Face Daily Papers & New Models:
{json.dumps(hf_candidates, indent=2)}

- Recent AI Leader & Researcher Interviews (with transcript excerpts):
{json.dumps(interview_candidates, indent=2)}

### 3. YOUR SYNTHESIS & ENGINEERING BREAKDOWN TASK
Using the live pre-fetched candidates above:
1. **Top AI News (Past 24 Hours):** Summarize 4-5 major releases, open-source drops, or infrastructure breakthroughs with direct source links from the news candidates above.
2. **AI Leader & Researcher Radar:** Analyze the pre-fetched interviews above to extract **emerging architectural patterns, agentic workflows, and upcoming tooling concepts** being discussed, and explain how an engineer can implement those concepts today.
3. **Standout
