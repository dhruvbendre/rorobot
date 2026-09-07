# rorobot

Roro, keeper of **Mnemora**: the archive world of Dhruv Bendre's solar-system portfolio. Clicking the Mnemora planet on the portfolio brings the visitor here. A visitor asks about Dhruv; Roro answers only from the documents the archive holds, cites them, and says plainly when it does not hold the answer.

Under the hood it is a small retrieval-augmented generation (RAG) pipeline behind a Streamlit page dressed as the interior of World VIII.

```
rorobot/
  app.py                 Streamlit entry point (the conversation)
  ingest.py              build / inspect the index from the command line
  config/settings.py     every setting, read from env or st.secrets
  rag/
    loader.py            markdown + frontmatter -> Document (placeholders excluded)
    chunker.py           heading-aware chunks with breadcrumbs
    embeddings.py        OpenAI embeddings, or a keyless local fallback
    vector_store.py      numpy cosine index, persisted to .index/
    retriever.py         hybrid top-k (embeddings + exact-word match) with a grounding floor
    prompts.py           Roro's voice and rules
    llm.py               openai | anthropic | none
    pipeline.py          Archive.ask(question, history) -> Answer
  ui/                    the world's look: Roro, horizon art, components
  styles/streamlit.css   the theme (hides default Streamlit chrome)
  knowledge/             the documents Roro answers from (edit these)
  tests/                 retrieval + refusal tests, no keys needed
  .streamlit/
    config.toml          base theme
    secrets.example.toml what to paste into Streamlit Cloud → Secrets
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (it is `dhruvbendre/rorobot`).
2. On https://share.streamlit.io choose **New app** → this repo, branch `main`, main file `app.py`.
3. Under **Advanced settings → Secrets** paste the contents of `.streamlit/secrets.example.toml` and fill in what you use. Nothing is required: with no keys Roro runs in no-model mode (it finds and quotes the right passages verbatim). Add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for answers in Roro's own words.
4. Set `PORTFOLIO_URL` to the portfolio's address so the **Back to the system** link works.
5. Deploy. Pick the app URL (for example `rorobot.streamlit.app`) and put that same URL in the portfolio's `VITE_ARCHIVE_URL` so the Mnemora planet points here.

The index is rebuilt from `knowledge/` on first start, so editing a markdown file and redeploying is all it takes to teach Roro something new.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS / Linux: cp .env.example .env
streamlit run app.py
```

Open http://localhost:8501. Python 3.11 or newer.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Thirteen tests cover loading, chunking, retrieval, refusal and the no-model answer path. None need an API key.

## Settings

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `openai`, `anthropic` or `none`. Blank auto-detects from the keys below. |
| `ANTHROPIC_API_KEY` | Anthropic chat (default model `claude-opus-5`). |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | OpenAI, or any OpenAI-compatible endpoint (Groq, xAI, Together, Ollama). |
| `MODEL_NAME` | Chat model override. |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | `openai` or `local`. Local is keyless and enough for this archive. |
| `RETRIEVAL_TOP_K`, `RETRIEVAL_MIN_SCORE`, `MAX_CONTEXT_CHARS`, `HISTORY_TURNS`, `CONDENSE_FOLLOWUPS`, `MAX_ANSWER_TOKENS` | Retrieval and answer limits. |
| `BOT_NAME`, `BOT_TAGLINE` | Who answers (Roro, keeper of the archive). |
| `ARCHIVE_NAME`, `OWNER_NAME`, `OWNER_SHORT`, `ARCHIVE_TAGLINE`, `ARCHIVE_EMPTY_STATE` | Names and copy. |
| `PORTFOLIO_URL` | Where **Back to the system** goes. |
| `KNOWLEDGE_DIR`, `INDEX_DIR`, `INCLUDE_TEST_DATA` | Storage. Leave blank on Streamlit Cloud. |

Every value can come from the environment, a local `.env`, or Streamlit's `st.secrets`, in that order.

## Adding knowledge

Drop a markdown file into `knowledge/` with a small frontmatter block (see the files already there). Sections become passages; a section whose body is only a `[PLACEHOLDER]` marker is skipped and counted as "still being written" in the footer. Rebuild locally with:

```bash
python ingest.py            # build the index
python ingest.py --check    # list documents and passages without embedding
```
