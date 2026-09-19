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
USER_ID = "alapan"


def extract_facts(user_message: str, assistant_reply: str) -> list[str]:
    """Our own lightweight extraction step — small, predictable prompt size."""
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
        # handle cases where model wraps JSON in ```json fences
        if text.startswith("```"):
            text = text.strip("`").replace("json\n", "", 1)
        facts = json.loads(text)
        if isinstance(facts, list):
            return [f for f in facts if isinstance(f, str)]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def chat(message: str) -> str:
    # 1. Retrieve relevant memories for this message
    relevant = memory.search(query=message, filters={"user_id": USER_ID}, limit=5)
    memory_text = "\n".join(f"- {m['memory']}" for m in relevant.get("results", []))

    system_prompt = "You are a helpful assistant."
    if memory_text:
        system_prompt += f"\n\nRelevant memories about the user:\n{memory_text}"

    # 2. Generate a response using those memories as context
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
    )
    reply = response.choices[0].message.content

    # 3. Extract facts ourselves (small, controlled prompt)
    facts = extract_facts(message, reply)

    # 4. Store each fact directly — infer=False means Mem0 just embeds & stores it,
    #    no internal LLM call, so no rate-limit risk here
    for fact in facts:
        memory.add(fact, user_id=USER_ID, infer=False)

    return reply


if __name__ == "__main__":
    print("Chat with your memory-enabled agent. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        reply = chat(user_input)
        print(f"Agent: {reply}\n")