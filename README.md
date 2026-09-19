# 🧠 Self-Learning AI Agent

A conversational AI agent with **long-term memory** — it remembers facts about you across a conversation and updates them intelligently when they change, instead of forgetting everything (or endlessly duplicating facts) the way most chatbots do.

**🔗 Live demo:** [self-learning-agent-svq4mu9nrdkhgg9tyqhapn.streamlit.app](https://self-learning-agent-svq4mu9nrdkhgg9tyqhapn.streamlit.app)

---

## The Problem

Most LLM-based chat apps are stateless — every message is processed in isolation with no memory of what came before, unless the *entire* conversation history is re-sent on every call (expensive, slow, and doesn't scale).

This project solves that with a lightweight **extract → categorize → store → retrieve** memory pipeline, so the agent recalls relevant facts about the user without re-sending full chat history — and correctly *updates* facts when they change, rather than accumulating contradictory memories over time.

## How It Works

1. **Chat** — the user sends a message; the agent generates a reply using an LLM (Groq)
2. **Extract** — a lightweight, separate LLM call pulls out standalone facts worth remembering from the exchange, each tagged with a fixed category key (`name`, `location`, `job`, `education`, `preference`, `goal`, `other`)
3. **Reconcile** — before storing a new fact, the agent checks for an existing memory with the *same category key* for that user. If one exists, it's deleted and replaced — so "I'm from Chennai" followed later by "actually I moved to Bangalore" correctly updates the location instead of leaving both facts stored and contradicting each other
4. **Store** — the fact is embedded locally and stored in a vector database (Qdrant)
5. **Retrieve** — on future messages, the agent semantically searches stored memories and injects relevant ones into context before responding

This means the agent only recalls what's *relevant* to the current message — not the entire conversation history — and its picture of the user stays internally consistent as facts change.

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| LLM | [Groq](https://groq.com) (`openai/gpt-oss-20b`) | Free tier, very fast inference |
| Memory layer | [Mem0](https://mem0.ai) (open-source) | Manages fact storage/retrieval |
| Vector store | [Qdrant](https://qdrant.tech) (local, embedded mode) | No server/Docker required |
| Embeddings | `sentence-transformers` (local, free) | No embedding API cost |
| UI | [Streamlit](https://streamlit.io) | Fast, clean chat interface, deployed on Streamlit Community Cloud |

**100% free to run** — no paid API keys required.

## Setup

```bash
git clone https://github.com/Alapan-Khan/self-learning-agent.git
cd self-learning-agent
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here


Get a free key at [console.groq.com](https://console.groq.com) — no credit card required.

## Running It

**Streamlit app (recommended):**
```bash
streamlit run streamlit_app.py
```

**CLI version:**
```bash
python memory_agent.py
```

## Project Structure

self-learning-agent/
├── agent_core.py # Core memory pipeline (chat, extract_facts, store_fact, get_all_memories, clear_all_memories)
├── memory_agent.py # CLI chat interface
├── streamlit_app.py # Web UI (deployed version)
├── requirements.txt
├── .streamlit/
│ └── config.toml # Disables Streamlit's file watcher (avoids a torchvision import crash on deploy)
└── .env # Not committed — holds your API key


## Design Decisions

- **Custom fact extraction instead of Mem0's default pipeline** — Mem0's built-in extraction prompt exceeded Groq's free-tier rate limit (8,000 tokens/minute). A smaller, purpose-built extraction prompt keeps requests well under that ceiling while still reliably capturing user facts.
- **Fixed-vocabulary category keys, not freeform tags** — early versions let the LLM invent its own category label per fact, which meant "location" facts sometimes got tagged `location`, sometimes `origin`, sometimes `hometown` — breaking deduplication. Constraining the model to a fixed set of keys makes conflict detection reliable.
- **Retry-on-empty-response** — Groq occasionally returns a blank completion for the extraction call under real usage. Instead of silently losing that fact, `extract_facts()` retries up to twice before giving up, which measurably improved reliability on the live deployment.
- **Local embeddings over a hosted embedding API** — keeps the entire stack free and removes a network dependency for a core part of the pipeline.
- **Qdrant in embedded/local mode** — no Docker or external server needed, making the project easy to clone and run immediately.
- **In-app "Clear Memory" button** — since Streamlit Cloud's filesystem can reset between deploys, a manual reset control makes the memory store easy to inspect and demo cleanly without needing file-system access.

## Known Limitations

- **Storage isn't guaranteed to persist across app restarts on Streamlit Cloud** — local Qdrant storage lives on the deployed container's disk, which can reset when the app reboots or sleeps. For a fully durable memory store, a hosted vector DB (Qdrant Cloud, Pinecone, etc.) would be a natural next step.
- **Single fixed user ID** — the deployed demo doesn't have per-visitor identity, so all visitors currently share one memory store.
- **Category keys are coarse** — a fact tagged `preference` could cover anything from food tastes to communication style; a more granular taxonomy (or letting Mem0's own conflict-resolution logic run, once rate limits allow it) would produce sharper deduplication.
