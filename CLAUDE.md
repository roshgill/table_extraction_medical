# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent PDF table extraction pipeline using Google Gemini (vision LLM). A routing agent identifies table types in PDF images, then dispatches to specialized extraction sub-agents.

## Repository Structure

```
HumaAI/
├── papers/                      # Paper-centric data (one dir per paper)
│   ├── ctt_lancet_2024/
│   │   ├── paper.pdf
│   │   └── ground_truth/        # p7_fig1_forest_plot.csv, etc.
│   ├── hodkinson_bmj_2022/
│   │   ├── paper.pdf
│   │   └── ground_truth/
│   └── cpy_document/
│       ├── paper.pdf
│       └── ground_truth/
├── agents/                      # One sub-package per extraction agent
│   ├── router/                  # Page classification (stub)
│   │   ├── prompt.py            # ROUTER_PROMPT string
│   │   └── classify.py          # classify_page() stub
│   ├── forest_plot/             # Forest plot table extraction
│   │   ├── prompt.py            # FOREST_PLOT_PROMPT string
│   │   ├── schema.py            # ForestPlotExtraction, PageForestPlotResult
│   │   └── extract.py           # extract_forest_plots_from_page(), stitch_forest_plot_results()
│   ├── general_table/           # General table extraction
│   │   ├── prompt.py            # GENERAL_TABLE_PROMPT string
│   │   ├── schema.py            # TableSegment, PageExtractionResult
│   │   └── extract.py           # extract_tables_from_page(), stitch_page_results()
│   └── simple_table/            # Simple table extraction (stub)
│       ├── prompt.py, schema.py, extract.py
│       └── (follows same pattern as general_table)
├── shared/                      # Code reused across all agents
│   ├── client.py                # Gemini client + DEFAULT_MODEL
│   ├── pdf.py                   # render_pages() — pdf2image wrapper
│   └── eval.py                  # normalize_text(), compare_tables(), print_accuracy_summary()
├── notebooks/                   # Eval and exploration notebooks
│   ├── eval_forest_plot.ipynb   # Catalog-driven forest plot eval
│   ├── eval_simple_table.ipynb  # Catalog-driven simple table eval
│   └── archive/                 # Old exploratory notebooks
├── table_catalog.csv            # Single index of all tables across all papers
├── output/                      # Generated at runtime (gitignored)
├── .env                         # GEMINI_API_KEY (gitignored)
└── requirements.txt
```

## Development Setup

```bash
pip install -r requirements.txt
# Run an eval notebook:
cd notebooks && jupyter notebook eval_forest_plot.ipynb
```

**Dependencies:** google-genai, python-dotenv, pdf2image, pandas, pdfplumber

## Key Patterns

- **Paper-centric data:** Each paper lives in `papers/<paper_id>/` with its PDF and ground truth CSVs
- **Catalog-driven eval:** `table_catalog.csv` is the single index — notebooks filter by agent and status to run evals
- **Agent modules** take `client` and `model` as function args (injected, not imported) for testability
- **Shared code** lives in `shared/` — client setup, PDF rendering, eval utilities
- **Each agent** has: `prompt.py` (prompt string), `schema.py` (Pydantic models), `extract.py` (core logic)
- **Notebooks** are thin eval drivers that import from modules — keep logic in `.py` files

## Naming Conventions

- **paper_id:** `<first_author_or_acronym>_<journal>_<year>` (e.g., `ctt_lancet_2024`)
- **Ground truth files:** `p<page>_<figure_id>_<descriptive_name>.csv`
- **Catalog columns:** paper_id, page, figure_id, table_type, agent, ground_truth_path, description, difficulty, status

## Adding a New Paper

1. Create `papers/<paper_id>/` with `paper.pdf` and `ground_truth/` dir
2. Add ground truth CSVs named `p<page>_<figure_id>_<type>.csv`
3. Add rows to `table_catalog.csv` for each table/figure

## Adding a New Agent

1. Create `agents/<name>/` with `__init__.py`, `prompt.py`, `schema.py`, `extract.py`
2. Add a corresponding `notebooks/eval_<name>.ipynb`
3. Update catalog entries to reference the new agent
