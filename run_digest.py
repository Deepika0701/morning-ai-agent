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


def run_digest_agent(
    history: dict,
    hf_candidates: list[dict],
    news_candidates: list[dict],
    tool_candidates: list[dict],
    interview_candidates: list[dict],
) -> dict:
    """Synthesizes the morning briefing using 100% Free-Tier Gemini models with retry & fallback."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
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
3. **Standout AI Tool / Engineering Concept of the Day:** Pick 1 high-impact NEW AI developer tool, framework, or repo from the tool/news candidates above NOT in `seen_history.tools`. Break down:
   - **Under the Hood:** Exact architecture and how it works technically.
   - **Quick Implementation Guide:** Concrete setup commands, CLI snippet, or minimal Python/config code showing how to wire it up.
   - **3 Real-World Engineering Use Cases:** Specific problems it solves, inputs/outputs, and when to use it vs. existing alternatives.
4. **Standout AI Model / Paper of the Day:** Pick 1 standout NEW model or architecture from the Hugging Face candidates above NOT in `seen_history.models`. Break down:
   - **Specs & Architecture:** How it works, parameter scale/efficiency, and key benchmark or capability highlights.
   - **How to Run / Integrate:** Concrete Hugging Face / Python / API invocation snippet showing how a developer would test or integrate it.
   - **3 Concrete Use Cases:** Specific production workflows where this model/technique shines.

### 4. REQUIRED OUTPUT FORMAT
Return ONLY a valid JSON object matching this exact schema:
{{
  "subject": "String: Crisp technical subject line naming today's featured tool/concept & model",
  "selected_model_id": "String: Unique ID or URL of the featured model to append to seen_history",
  "selected_tool_id": "String: Unique repo name or URL of the featured tool to append to seen_history",
  "selected_interview_id": "String: Video ID or URL of the featured interview to append to seen_history",
  "html_body": "String: Clean, responsive, inline-CSS styled HTML email (max-width 680px, readable code blocks with monospace font & dark/light contrast, clear headings) containing these 4 sections:
    1. ⚡ 2-Minute Daily AI News Flash (4-5 crisp bullets with source links)
    2. 🎙️ AI Leader Radar: Emerging Concepts & How to Apply Them
    3. 🛠️ Deep Dive: Tool / Concept Implementation & Use Cases (Architecture + Code/CLI Quickstart + 3 Concrete Use Cases)
    4. 🧠 Deep Dive: Standout Model Implementation & Use Cases (Specs + Run/API Snippet + 3 Concrete Use Cases)"
}}
"""

    # 100% Free-Tier Gemini model cascade with retry on transient 503 overload
    free_tier_models = [
        "gemini-3.8-flash",
        "gemini-3.8-flash",       # Retry once after backoff if 503 spike occurs
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]

    for attempt_idx, model_name in enumerate(free_tier_models):
        try:
            print(f"Attempt {attempt_idx + 1}: Querying {model_name} (Free Tier)...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            raw_text = response.text.strip()
            match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            cleaned = match.group(0) if match else raw_text
            return json.loads(cleaned)
        except Exception as exc:
            print(f"[Warn] {model_name} attempt failed: {exc}")
            time.sleep(15)

    raise RuntimeError("All Free-Tier Gemini attempts failed.")


def send_email(subject: str, html_body: str) -> None:
    sender = os.environ["GMAIL_USER"].strip()
    raw_recipients = os.environ.get("RECIPIENT_EMAIL", sender)
    app_password = os.environ["GMAIL_APP_PASSWORD"].strip()

    recipient_list = [
        addr.strip()
        for addr in re.split(r"[,;\s\n]+", raw_recipients)
        if addr.strip() and "@" in addr
    ]

    plain_fallback = re.sub(r"<[^>]+>", " ", html_body)
    plain_fallback = re.sub(r"\s+", " ", plain_fallback).strip()

    print(f"Sending digest to {len(recipient_list)} recipient(s): {recipient_list}")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        for recipient in recipient_list:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = f"Morning AI Agent <{sender}>"
            msg["To"] = recipient
            msg["Date"] = formatdate(localtime=True)
            msg["Message-ID"] = make_msgid(domain="gmail.com")

            msg.attach(MIMEText(plain_fallback, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            server.sendmail(sender, [recipient], msg.as_string())
            print(f" -> Delivered to: {recipient}")


def main() -> None:
    history = load_history()
    hf_candidates = fetch_unseen_hf_models(history.get("models", []))
    news_candidates = fetch_rss_items(AI_NEWS_FEEDS, history.get("tools", []), limit=10)
    tool_candidates = fetch_rss_items(AI_TOOLS_FEEDS, history.get("tools", []), limit=10)
    interview_candidates = fetch_unseen_interviews(history.get("interviews", []))

    result = run_digest_agent(
        history, hf_candidates, news_candidates, tool_candidates, interview_candidates
    )
    send_email(result["subject"], result["html_body"])

    if result.get("selected_model_id"):
        history.setdefault("models", []).append(result["selected_model_id"])
    if result.get("selected_tool_id"):
        history.setdefault("tools", []).append(result["selected_tool_id"])
    if result.get("selected_interview_id"):
        history.setdefault("interviews", []).append(result["selected_interview_id"])

    save_history(history)
    print("Updated seen_history.json successfully.")


if __name__ == "__main__":
    main()
