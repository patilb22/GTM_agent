# GTM Agent Starter Kit

A LangChain + LangGraph starter for a GTM content-suite agent that uses RAG, review, checkpointers, interrupts, and LangSmith tracing.

## Files
- `data/products.csv` — synthetic product list
- `data/events.md` — synthetic event calendar / Notion export style
- `data/past_campaigns.md` — synthetic past messaging library
- `src/gtm_agent.py` — LangGraph workflow

## Run
```bash
pip install langchain langgraph langchain-openai langchain-community langchain-chroma chromadb pydantic streamlit pandas
export OPENAI_API_KEY=
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=
cd gtm_agent_starter
python src/gtm_agent.py
```

PowerShell:
```powershell
$env:OPENAI_API_KEY="your_key_here"
$env:LANGSMITH_TRACING="true"
$env:LANGSMITH_API_KEY="your_langsmith_key_here"
cd gtm_agent_starter
python src/gtm_agent.py
```

## Streamlit App
```bash
cd gtm_agent_starter
streamlit run streamlit_app.py
```

If Streamlit is already running, stop it first with `Ctrl+C`, set `$env:OPENAI_API_KEY`, then start it again. Environment variables set after Streamlit starts are not picked up by the running server.
