# Papers

Each paper gets its own directory with the PDF and ground truth CSVs.

## Directory Structure

```
papers/<paper_id>/
├── paper.pdf                  # Source PDF
└── ground_truth/              # One CSV per table/figure
    └── p<page>_<figure_id>_<type>.csv
```

## Papers Index

| paper_id | Full Title | Tables |
|---|---|---|
| `ctt_lancet_2024` | Effects of Statins — Lancet (CTT Collaboration) | Forest plots (p7, p9), characteristics table (p5) |
| `hodkinson_bmj_2022` | Hodkinson et al. BMJ 2022 (network meta-analysis) | Simple table (p3), forest plot (p5), league table (p6) |
| `cpy_document` | CPY Clinical Document | Principal investigators (p49-51), severity scores (p31), non-inferiority analyses (p31) |

## Naming Convention

Ground truth files follow: `p<page>_<figure_id>_<descriptive_name>.csv`

Examples:
- `p7_fig1_forest_plot.csv` — Forest plot from page 7, figure 1
- `p31_table1_investigator_global_severity.csv` — Table 1 on page 31
