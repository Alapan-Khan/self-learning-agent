import os
import json
from dotenv import load_dotenv
from groq import Groq
from mem0 import Memory

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

config = {
    "llm": {
        "provider": "groq",
        "config": {
            "model": "openai/gpt-oss-20b",
            "api_key": os.getenv("GROQ_API_KEY"),
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
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    text = response.choices[0].message.content.strip()
    try:
        if text.startswith("```"):
            text = text.strip("`").replace("json\n", "", 1)
        facts = json.loads(text)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, str)]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def chat(message: str, user_id: str) -> str:
    """One full turn: retrieve relevant memories, generate a reply,
    extract new facts, store them."""
    relevant = memory.search(query=message, filters={"user_id": user_id}, limit=5)
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

    for fact in extract_facts(message, reply):
        memory.add(fact, user_id=user_id, infer=False)

    return reply


def get_all_memories(user_id: str) -> list[str]:
    """Fetch every stored memory for a user — used to show what the agent knows."""
    results = memory.get_all(filters={"user_id": user_id})
    return [m["memory"] for m in results.get("results", [])]