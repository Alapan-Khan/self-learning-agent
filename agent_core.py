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


def extract_facts(user_message: str, assistant_reply: str) -> list[str]:
    """Lightweight, controlled-size fact extraction (avoids Mem0's default
    extraction prompt, which is too large for Groq's free-tier TPM limit)."""
    prompt = (
        "Extract short standalone facts about the user worth remembering "
        "long-term (name, role, preferences, decisions). "
        'Return ONLY a JSON list of strings, e.g. ["User\'s name is X"]. '
        "If nothing is worth remembering, return [].\n\n"
        f"User: {user_message}\nAssistant: {assistant_reply}"
    )
    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        print(f"[extract_facts] raw LLM output: {text!r}", flush=True)

        if text.startswith("```"):
            text = text.strip("`").replace("json\n", "", 1)
        facts = json.loads(text)
        if isinstance(facts, list):
            result = [f for f in facts if isinstance(f, str)]
            print(f"[extract_facts] parsed facts: {result}", flush=True)
            return result
    except Exception as e:
        print(f"[extract_facts] FAILED: {type(e).__name__}: {e}", flush=True)
    return []


def chat(message: str, user_id: str) -> str:
    """One full turn: retrieve relevant memories, generate a reply,
    extract new facts, store them."""
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
        try:
            result = memory.add(fact, user_id=user_id, infer=False)
            print(f"[chat] stored fact: {fact!r} -> {result}", flush=True)
        except Exception as e:
            print(f"[chat] STORE FAILED for {fact!r}: {type(e).__name__}: {e}", flush=True)

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