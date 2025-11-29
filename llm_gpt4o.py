# llm_gpt4o.py
"""
Azure OpenAI API helper for counterfactual generation
"""
import os
import configparser
from openai import AzureOpenAI

# Try to read from config.ini first, then fall back to environment variables
config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
config.read('config.ini')

# Get Azure OpenAI settings (config.ini takes precedence over env vars)
if config.has_section('azure_openai'):
    API_KEY = config.get('azure_openai', 'api_key')
    AZURE_ENDPOINT = config.get('azure_openai', 'azure_endpoint')
    MODEL_ID = config.get('azure_openai', 'model')
    API_VERSION = config.get('azure_openai', 'api_version', fallback='2024-08-01-preview')
else:
    # Fall back to environment variables
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    MODEL_ID = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-2024-11-20")
    API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")


def get_client(timeout=600.0):
    """Initialize Azure OpenAI client"""
    if not API_KEY:
        raise ValueError(
            "Azure OpenAI API key not found. "
            "Add it to config.ini under [azure_openai] section or set AZURE_OPENAI_API_KEY environment variable"
        )
    if not AZURE_ENDPOINT:
        raise ValueError(
            "Azure OpenAI endpoint not found. "
            "Add it to config.ini under [azure_openai] section or set AZURE_OPENAI_ENDPOINT environment variable"
        )

    return AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        timeout=timeout
    )


def warmup(client):
    """Warmup call to ensure API is responsive"""
    try:
        client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
            temperature=0
        )
        print(f"Azure OpenAI warmup successful (Model: {MODEL_ID})")
    except Exception as e:
        print(f"Warmup warning: {e}")


def chat_call(client, messages, model=None, max_tokens=256, temperature=0.0):
    """Make a chat completion call"""
    if model is None:
        model = MODEL_ID

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()