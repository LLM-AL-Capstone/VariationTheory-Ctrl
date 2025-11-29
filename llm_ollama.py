# llm_ollama.py
import time, configparser
from typing import List, Dict

def _read_cfg():
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',';'))
    cfg.read('config.ini')
    get = lambda k, fb='': cfg.get('settings', k, fallback=fb).strip()
    return {
        "base_url": get('local_base_url', 'http://localhost:11434/v1') or 'http://localhost:11434/v1',
        "api_key":  get('local_api_key',  'ollama') or 'ollama',
        "model_id": get('model_id',       'qwen2:1.5b') or 'qwen2:1.5b',
    }

CFG = _read_cfg()
MODEL_ID = CFG["model_id"]

def get_client(timeout: float = 600.0):
    try:
        from openai import OpenAI
    except Exception:
        raise SystemExit("ERROR: `pip install openai` first.")
    client = OpenAI(api_key=CFG["api_key"], base_url=CFG["base_url"], timeout=timeout, max_retries=0)
    try:
        _ = client.models.list()
    except Exception as e:
        raise SystemExit(f"FATAL: Cannot reach Ollama at {CFG['base_url']} — {e}")
    return client

def warmup(client) -> None:
    try:
        client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role":"user","content":"OK"}],
            temperature=0, max_tokens=4
        )
        print("LOG: LLM warm-up done.")
    except Exception as e:
        print("WARN: warm-up failed:", e)

def chat_call(client, messages: List[Dict], model: str = MODEL_ID, max_tokens: int = 64, temperature: float = 0.0):
    """Small backoff to ride out transient CPU stalls."""
    delays = [0, 8, 16]
    last = None
    for d in delays:
        if d: time.sleep(d)
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens, stop=["\n"]
            )
            return r.choices[0].message.content
        except Exception as e:
            last = e
            print("WARN: chat attempt failed:", e)
    raise last
