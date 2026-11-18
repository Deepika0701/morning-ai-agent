# Morning AI Digest — free setup

## What this package does

- Reads public RSS feeds and Hugging Face Daily Papers (no source API keys required).
- Reads public YouTube channel feeds and attempts to fetch English captions (no YouTube API key required; captions are not guaranteed to be available).
- Uses OpenRouter **only** when the discovered model ID ends with `:free`.
- Optionally uses a local Ollama model on the same machine, with no hosted inference API charge.
- Sends the resulting email through Gmail SMTP using a Google App Password.
- Saves separate histories for news, tools, papers, and interviews.

There is no Groq, Gemini, GitHub Models, paid OpenRouter model, web-search API, or paid scraping service in this version.

## Install locally

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`, replace the placeholders, then load it in your shell:

```bash
cp .env.example .env
# Edit .env, then:
set -a
source .env
set +a
python run_digest.py
```

The script does not load `.env` automatically; the `source` command exports those variables to the process.

## Configure the free hosted model

1. Create an OpenRouter account and API key.
2. Put the key in `OPENROUTER_API_KEY`.
3. The script queries the live model catalog and calls only model IDs whose names end in `:free`. It intentionally ignores other zero-priced IDs because some may require a funded account.
4. If every free model is unavailable or rate-limited, it tries the optional local Ollama model. If that is also unavailable, it still emails a source-only digest and clearly labels it as such.

Free model availability and quotas can change. OpenRouter currently documents 50 requests/day for its free tier; the script tries at most three model IDs in one run.

## Gmail setup

1. Enable 2-Step Verification on the Google account.
2. Create a Google App Password for this script.
3. Set `GMAIL_USER` and `GMAIL_APP_PASSWORD`.
4. Set `RECIPIENT_EMAIL` if you want delivery to a different address or multiple recipients.

Use an App Password, not your normal Gmail password. Never commit `.env` or paste secrets into source code.

## Run daily with GitHub Actions

Copy `.github/workflows/morning-digest.yml` into the same path in your repository. In **Settings → Secrets and variables → Actions**, add:

- `GMAIL_USER`
- `GMAIL_APP_PASSWORD`
- `RECIPIENT_EMAIL` (optional; defaults to `GMAIL_USER`)
- `OPENROUTER_API_KEY` (optional; without it the email uses the source-only fallback unless local Ollama is configured on the runner)

The workflow is scheduled for 00:30 UTC (06:00 in Asia/Kolkata) and can also be started manually. It commits `seen_history.json` after successful delivery so the next run remembers what it has already covered.

Standard GitHub-hosted runners are free for public repositories. Private repositories have a plan-dependent included monthly allowance; usage beyond that allowance may be billable. Check your GitHub plan's Actions allowance before relying on a private repository for a zero-cost setup.

## Optional local Ollama

When running the Python script on your own computer, you can use a model already downloaded in Ollama:

```bash
ollama pull qwen2.5:7b
ollama serve
export OLLAMA_MODEL=qwen2.5:7b
python run_digest.py
```

This uses your computer's resources rather than a hosted inference API. It will not work from GitHub-hosted runners unless Ollama is installed and a model is available on that runner.

## Check syntax

```bash
python -m py_compile run_digest.py
```
