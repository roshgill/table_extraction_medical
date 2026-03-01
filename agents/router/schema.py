"""Pydantic schemas for router page classification."""

from pydantic import BaseModel


class TableClassification(BaseModel):
    label: str          # e.g. "Table 1", "Figure 2A", "top forest plot"
    type: str           # "forest_plot" or "general_table"
    description: str    # one sentence
    instruction: str    # extraction instruction for the sub-agent


class PageClassification(BaseModel):
    tables: list[TableClassification]
