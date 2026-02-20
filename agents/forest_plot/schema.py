"""Pydantic schemas for forest plot extraction."""

from pydantic import BaseModel


class ForestPlotExtraction(BaseModel):
    title: str
    headers: list[str]
    rows: list[list[str]]
    footnotes: list[str]
    plot_appears_complete: bool


class PageForestPlotResult(BaseModel):
    forest_plots: list[ForestPlotExtraction]
