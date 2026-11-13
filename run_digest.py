import json
import os
import re
import smtplib
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import feedparser
import requests
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi

HISTORY_FILE = Path("seen_history.json")
MAX_HISTORY_ITEMS = 60

# High-signal AI Leader & Researcher Interview Channels (YouTube Channel IDs)
INTERVIEW_CHANNELS = {
    "Dwarkesh Patel": "UCXl4i9dYBrFOabk0xGmbkRA",
    "Latent Space": "UCcH12M3qp5C9hB2U8U1Z3gQ",
    "No Priors (Sarah Guo & Elad Gil)": "UCn8zNIfYAQNdrFRrr8oibKw",
    "Lex Fridman": "UCSHZKyawb77ixDdsGog4iWA",
    "Y Combinator": "UCcefcZRL2oaA_uBNeo5UOWg",
}


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

                transcript_snippet = "Transcript unavailable; rely on web search & description."
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                    # Grab a dense slice of the conversation (skipping the first 2 mins of ads/intro)
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


def run_antigravity_agent(history: dict, hf_candidates: list[dict], interview_candidates: list[dict]) -> dict:
    """Runs the Antigravity Managed Agent in a remote sandbox and returns structured JSON."""
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options={"api_version": "v1beta"},
    )

    today_str = datetime.now(timezone.utc).strftime("%B %d, %Y")

    prompt = f"""
You are an elite AI Intelligence Agent preparing my morning AI briefing for {today_str}.

### 1. ALREADY SEEN LEDGER (DO NOT REPEAT THESE IN DEEP DIVES)
{json.dumps(history, indent=2)}

### 2. PRE-FETCHED UNSEEN CANDIDATES FROM TODAY
- Hugging Face Papers/Models Candidates:
{json.dumps(hf_candidates, indent=2)}

- Recent AI Leader/Founder Interview Candidates (with transcript excerpts):
{json.dumps(interview_candidates, indent=2)}

### 3. YOUR RESEARCH TASK
Use your built-in `google_search` and `url_context` tools to:
1. Gather the top 4-5 breaking AI news stories from the past 24 hours.
2. Analyze the pre-fetched AI Leader Interviews (or search for any brand-new interview from Sam Altman, Dario Amodei, Demis Hassabis, Yann LeCun, Jensen Huang, Karpathy, etc. from the last 48 hours not in the seen ledger) to extract what upcoming tools, bottlenecks, and technical shifts they are telegraphing.
3. Pick 1 standout NEW AI Developer Tool (from GitHub Trending AI, Product Hunt AI, or major launches in the last 48 hours) that is NOT in `seen_history.tools`. Inspect its repo/docs via `url_context`.
4. Pick 1 standout NEW AI Model (either from the HF candidates above or a major frontier/open-weights launch from the last 48 hours) that is NOT in `seen_history.models`.

### 4. REQUIRED OUTPUT FORMAT
Return ONLY a valid JSON object (no markdown fences outside the JSON) matching this exact schema:
{{
  "subject": "String: Engaging email subject line including today's standout model & tool names",
  "selected_model_id": "String: Unique ID or URL of the featured model to append to seen_history",
  "selected_tool_id": "String: Unique repo name or URL of the featured tool to append to seen_history",
  "selected_interview_id": "String: Video ID or URL of the featured interview to append to seen_history",
  "html_body": "String: Complete, inline-CSS styled HTML email containing these 4 sections:
    1. ⚡ 2-Minute Daily AI News Flash (4-5 crisp bullets with source links)
    2. 🎙️ AI Leader & Founder Radar (Who spoke, key quotes/signals, and what upcoming tools/paradigms this predicts)
    3. 🛠️ Deep Dive: Standout AI Tool of the Day (What it is, how it works under the hood, and 3 concrete engineering use cases)
    4. 🧠 Deep Dive: Standout Model of the Day (Architecture/benchmarks, weights/API availability, and 3 concrete use cases)"
}}
"""

    interaction = client.interactions.create(
        agent="antigravity-preview-09-2026",
        input=prompt,
        background=True,
        environment={"type": "remote"},
    )

    print(f"Started Antigravity interaction: {interaction.id}")
    while interaction.status in ("in_progress", "requires_action"):
        time.sleep(10)
        interaction = client.interactions.get(interaction.id)
        print(f"Status: {interaction.status}")

    if interaction.status != "completed":
        raise RuntimeError(f"Antigravity agent failed with status: {interaction.status}")

    raw_text = interaction.output_text.strip()
    # Extract the JSON object cleanly even if wrapped in markdown fences or extra prose
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    cleaned = match.group(0) if match else raw_text
    return json.loads(cleaned)


def send_email(subject: str, html_body: str) -> None:
    sender = os.environ["GMAIL_USER"]
    raw_recipients = os.environ.get("RECIPIENT_EMAIL", sender)
    app_password = os.environ["GMAIL_APP_PASSWORD"]

    # Split comma-separated emails into a clean Python list
    recipient_list = [email.strip() for email in raw_recipients.split(",") if email.strip()]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Morning AI Agent <{sender}>"
    msg["To"] = ", ".join(recipient_list)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, recipient_list, msg.as_string())
    print(f"Email successfully delivered to: {', '.join(recipient_list)}")

def main() -> None:
    history = load_history()
    hf_candidates = fetch_unseen_hf_models(history.get("models", []))
    interview_candidates = fetch_unseen_interviews(history.get("interviews", []))

    result = run_antigravity_agent(history, hf_candidates, interview_candidates)
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
