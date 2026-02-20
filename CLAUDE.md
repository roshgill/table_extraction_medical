# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent PDF table extraction pipeline using Google Gemini (vision LLM). A routing agent identifies table types in PDF images, then dispatches to specialized extraction sub-agents.

## Repository Structure

```
HumaAI/
├── agents/                      # One sub-package per extraction agent
│   ├── forest_plot/             # Forest plot table extraction
│   │   ├── prompt.py            # FOREST_PLOT_PROMPT string
│   │   ├── schema.py            # ForestPlotExtraction, PageForestPlotResult
│   │   ├── extract.py           # extract_forest_plots_from_page(), stitch_forest_plot_results()
│   │   ├── test_data/           # Agent-specific PDFs + ground truth
│   │   └── notebook.ipynb       # Thin driver: config → run → eval
│   └── general_table/           # General table extraction (same pattern)
│       ├── prompt.py, schema.py, extract.py
│       ├── test_data/
│       └── notebook.ipynb
├── shared/                      # Code reused across all agents
│   ├── client.py                # Gemini client + DEFAULT_MODEL
│   ├── pdf.py                   # render_pages() — pdf2image wrapper
│   └── eval.py                  # normalize_text(), compare_tables(), print_accuracy_summary()
├── notebooks/                   # Archived exploratory notebooks
├── example sources/             # Source PDFs
├── output/                      # Generated at runtime (gitignored)
├── table_catalog.csv
├── .env                         # GEMINI_API_KEY (gitignored)
└── requirements.txt
```

## Development Setup

```bash
pip install -r requirements.txt
# Run an agent notebook:
cd agents/forest_plot && jupyter notebook notebook.ipynb
```

**Dependencies:** google-genai, python-dotenv, pdf2image, pandas, pdfplumber

## Key Patterns

- **Agent modules** take `client` and `model` as function args (injected, not imported) for testability
- **Shared code** lives in `shared/` — client setup, PDF rendering, eval utilities
- **Each agent** has: `prompt.py` (prompt string), `schema.py` (Pydantic models), `extract.py` (core logic)
- **Notebooks** are thin drivers that import from modules — keep logic in `.py` files
- **Test data** lives in each agent's `test_data/` dir (PDFs symlinked from `example sources/`)

## Adding a New Agent

1. Create `agents/<name>/` with `__init__.py`, `prompt.py`, `schema.py`, `extract.py`
2. Add `test_data/` with PDFs and ground truth CSVs
3. Add `notebook.ipynb` importing from `shared/` and the agent module
