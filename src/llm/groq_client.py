"""Step 6.1: small Groq chat client with a disk cache and rate-limit handling.

Groq serves open models through an OpenAI-style web API. We send a list of messages
(system = rules, user = the task) and get the model's reply back.

- Cache: every reply is saved in data/processed/llm/cache/, keyed by the full request
  (model, messages, settings, sample number). Same request again = read from disk:
  reproducible, and no tokens spent twice.
- Rate limits (free plan): ~8,000 tokens per minute per model. When Groq answers 429
  ("too many requests") we wait as long as it says (retry-after) and try again.
- The API key is read with os.getenv("GROQ_API_KEY") from .env and is never printed or cached.
"""
import hashlib
import json
import os
import time

import requests
from dotenv import load_dotenv

from src import config

URL = "https://api.groq.com/openai/v1/chat/completions"
CACHE_DIR = config.PROCESSED_DIR / "llm" / "cache"
load_dotenv(config.ROOT / ".env")


class RateLimitDaily(RuntimeError):
    """Raised when Groq says the DAILY limit is used up (waiting minutes will not help)."""


def chat(model, messages, temperature=0.0, max_tokens=2000, json_mode=True, sample=0, **extra):
    """Reply text (and usage) for one chat request. `sample` > 0 asks for a new, separately cached reply."""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, **extra}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    key = hashlib.sha1(json.dumps([body, sample], sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}
    r = None
    for attempt in range(8):
        try:
            r = requests.post(URL, headers=headers, json=body, timeout=180)
        except (requests.ConnectionError, requests.Timeout) as e:   # Wi-Fi drop, sleep, DNS failure
            wait = min(300, 15 * 2 ** attempt)
            print(f"  network error ({type(e).__name__}), retrying in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            break
        if r.status_code == 429:
            msg = r.text.lower()
            if "per day" in msg or "tpd" in msg or "rpd" in msg:
                raise RateLimitDaily(r.json().get("error", {}).get("message", "daily limit reached"))
            wait = float(r.headers.get("retry-after", 10))
            print(f"  rate limit, waiting {wait:.0f}s")
            time.sleep(wait + 1)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()                          # 400 etc.: a bug in our request, stop
    if r is None:
        raise ConnectionError("no network connection to Groq after several retries")
    r.raise_for_status()

    data = r.json()
    out = {"model": model, "text": data["choices"][0]["message"]["content"], "usage": data.get("usage", {}),
           "finish_reason": data["choices"][0].get("finish_reason")}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out
