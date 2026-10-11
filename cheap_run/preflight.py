"""
Preflight check run before any experiment is launched.

Why this exists: an invalid key (HTTP 401) or an empty credit balance
(HTTP 400) does NOT crash a ShinkaEvolve run. Generation 0 needs no model call
and scores normally, so a run can look healthy while making zero progress.
This makes one tiny request per provider and refuses to continue unless it
succeeds. It never prints a key.
"""
import json
import os
import re
import sys
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_env(name):
    val = os.environ.get(name)
    path = os.path.join(REPO, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8-sig", errors="ignore"):
            m = re.match(rf"\s*{name}\s*=\s*(.*?)\s*$", line)
            if m and m.group(1):
                val = m.group(1).strip().strip('"').strip("'")
    return val


def post(url, headers, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        urllib.request.urlopen(req, timeout=30)
        return 200, ""
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", {}).get("message", "")
        except Exception:
            msg = ""
        return e.code, msg
    except Exception as e:  # network down etc.
        return -1, f"{type(e).__name__}: {e}"


problems = []

akey = read_env("ANTHROPIC_API_KEY")
if not akey:
    problems.append("ANTHROPIC_API_KEY is missing from .env")
else:
    for model in ("claude-sonnet-5", "claude-haiku-4-5-20251001"):
        code, msg = post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": akey, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        )
        print(f"  Anthropic {model}: " + ("OK" if code == 200 else f"HTTP {code} {msg[:110]}"))
        if code == 401:
            problems.append("Anthropic key rejected (401): create a new key and put it in .env")
            break
        if code == 400 and "credit" in msg.lower():
            problems.append("Anthropic credit balance is empty: add credit at console.anthropic.com -> Plans & Billing")
            break
        if code not in (200,):
            problems.append(f"Anthropic call to {model} failed (HTTP {code}): {msg[:100]}")

okey = read_env("OPENAI_API_KEY")
if not okey:
    problems.append("OPENAI_API_KEY is missing from .env (needed for the novelty-check embeddings)")
else:
    code, msg = post(
        "https://api.openai.com/v1/embeddings",
        {"Authorization": f"Bearer {okey}", "content-type": "application/json"},
        {"model": "text-embedding-3-small", "input": "hi"},
    )
    print("  OpenAI embeddings: " + ("OK" if code == 200 else f"HTTP {code} {msg[:110]}"))
    if code != 200:
        problems.append(f"OpenAI embeddings call failed (HTTP {code}): {msg[:100]}")

if problems:
    print("\nPREFLIGHT FAILED:")
    for p in problems:
        print("  - " + p)
    sys.exit(1)
print("Preflight OK.")
