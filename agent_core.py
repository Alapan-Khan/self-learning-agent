import os
import json
from dotenv import load_dotenv
from groq import Groq
from mem0 import Memory
import streamlit as st

load_dotenv()


def get_api_key():
    """Reads the Groq API key from a local .env file (for local dev)
    or from Streamlit Cloud's secrets store (for deployment)."""
    return os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")


groq_client = Groq(api_key=get_api_key())

config = {
    "llm": {
        "provider": "groq",
        "config": {
            "model": "openai/gpt-oss-20b",
            "api_key": get_api_key(),
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "multi-qa-MiniLM-L6-cos-v1"
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "agent_memory",
            "path": "qdrant_data",
            "embedding_model_dims": 384
        }
    }
}

memory = Memory.from_config(config)

# Fixed vocabulary so the same type of fact always maps to the same key,
# letting new facts correctly replace old ones instead of duplicating.
ALLOWED_KEYS = ["name", "location", "job", "education", "preference", "goal", "other"]


def extract_facts(user_message: str, assistant_reply: str) -> list[dict]:
    """Extract facts as {key, value} pairs from a fixed vocabulary of keys,
    so conflicting facts (same key) can replace old ones instead of piling up."""
    prompt = (
        "Extract short standalone facts about the user worth remembering long-term. "
        "For each fact, assign it ONE of these exact category keys: "
        f'{json.dumps(ALLOWED_KEYS)}. '
        "Use the SAME key every time the same type of fact appears — e.g., always "
        '"location" for where the user lives or is originally from, even if phrased '
        "differently across messages. "
        "Return ONLY a JSON list like: "
        '[{"key": "location", "value": "Originally from West Bengal, now working in Bengaluru"}]. '
        "If nothing is worth remembering, return [].\n\n"
        f"User: {user_message}\nAssistant: {assistant_reply}"
    )
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        text = response.choices[0].message.content.strip()
        print(f"[extract_facts] raw LLM output: {text!r}", flush=True)

        if text.startswith("```"):
            text = text.strip("`").replace("json\n", "", 1)
        facts = json.loads(text)
        if isinstance(facts, list):
            result = [
                f for f in facts
                if isinstance(f, dict)
                and f.get("key") in ALLOWED_KEYS
                and f.get("value")
            ]
            print(f"[extract_facts] parsed facts: {result}", flush=True)
            return result
    except Exception as e:
        print(f"[extract_facts] FAILED: {type(e).__name__}: {e}", flush=True)
    return []


def store_fact(key: str, value: str, user_id: str):
    """Store a fact, replacing any existing memory with the same key
    (so 'location' updates in place instead of duplicating)."""
    try:
        existing = memory.get_all(filters={"user_id": user_id})
        for m in existing.get("results", []):
            if m.get("metadata", {}).get("key") == key:
                memory.delete(memory_id=m["id"])
                print(f"[store_fact] replaced old memory for key={key!r}", flush=True)

        result = memory.add(value, user_id=user_id, infer=False, metadata={"key": key})
        print(f"[store_fact] stored key={key!r} value={value!r} -> {result}", flush=True)
    except Exception as e:
        print(f"[store_fact] FAILED for key={key!r}: {type(e).__name__}: {e}", flush=True)


def chat(message: str, user_id: str) -> str:
    """One full turn: retrieve relevant memories, generate a reply,
    extract new facts, store/update them."""
    try:
        relevant = memory.search(query=message, filters={"user_id": user_id}, limit=5)
        print(f"[chat] search results: {relevant}", flush=True)
    except Exception as e:
        print(f"[chat] SEARCH FAILED: {type(e).__name__}: {e}", flush=True)
        relevant = {}

    memory_text = "\n".join(f"- {m['memory']}" for m in relevant.get("results", []))

    system_prompt = "You are a helpful assistant."
    if memory_text:
        system_prompt += f"\n\nRelevant memories about the user:\n{memory_text}"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
    )
    reply = response.choices[0].message.content

    facts = extract_facts(message, reply)
    print(f"[chat] facts to store: {facts}", flush=True)

    for fact in facts:
        store_fact(fact["key"], fact["value"], user_id)

    return reply


def get_all_memories(user_id: str) -> list[str]:
    """Fetch every stored memory for a user — used to show what the agent knows."""
    try:
        results = memory.get_all(filters={"user_id": user_id})
        print(f"[get_all_memories] raw results: {results}", flush=True)
        return [m["memory"] for m in results.get("results", [])]
    except Exception as e:
        print(f"[get_all_memories] FAILED: {type(e).__name__}: {e}", flush=True)
        return []