# MMM Budget Allocation Agent

This project is a simple marketing mix modeling (MMM) and budget optimization app.

The workflow is intentionally small and easy to explain:

1. Upload marketing spend data
2. Confirm the schema
3. Run backward analysis
4. Optimize the budget across channels

It uses a local or cloud LLM only as a guide for analysis and reporting, while the real optimization logic is handled by the Python pipeline.

## Main folders

- `app/` — Streamlit UI pages and entry point
- `src/` — data prep, MMM logic, optimizer, agent, guardrails
- `tests/` — small regression tests
- `data/sample/` — demo dataset
- `config.yaml` — default app settings
- `requirements.txt` — Python dependencies
- `.env.example` — environment variables for model providers

## Run locally

```bash
pip install -r requirements.txt
copy .env.example .env
streamlit run app/app.py
```

Set the model provider in `.env`:

- Local Ollama: `OLLAMA_MODEL=llama3.1:8b`
- Optional Anthropic or Hugging Face keys if you want cloud models

## Example flow

```text
Upload dataset -> Confirm columns -> Backward Analysis -> Allocation
```

That is the full product story in one line.
