"""Forest plot extraction logic."""

import pandas as pd
from PIL import Image

from .prompt import FOREST_PLOT_PROMPT
from .schema import PageForestPlotResult


def extract_forest_plots_from_page(
    client,
    model: str,
    image: Image.Image,
    continuation_context: dict | None = None,
) -> PageForestPlotResult:
    """Send a page image to Gemini and extract forest plot tables.

    Args:
        client: google.genai.Client instance.
        model: Model name string.
        image: PIL image of the page.
        continuation_context: dict with 'title' and 'headers'
            from the previous page's incomplete forest plot, or None.
    """
    prompt = FOREST_PLOT_PROMPT
    if continuation_context:
        prompt += f"""\n\nNOTE: The first forest plot on this page is a CONTINUATION from the previous page.
Title: {continuation_context['title']}
Headers: {continuation_context['headers']}

For that forest plot, use the title and headers above (return them as-is).
Each row must have exactly {len(continuation_context['headers'])} values.
Any additional forest plots after it on this page should be treated as new entries."""

    response = client.models.generate_content(
        model=model,
        contents=[prompt, image],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": PageForestPlotResult.model_json_schema(),
        },
    )

    return PageForestPlotResult.model_validate_json(response.text)


def stitch_forest_plot_results(
    page_results: dict[int, PageForestPlotResult],
) -> list[tuple[str, pd.DataFrame]]:
    """Merge page results into DataFrames — one per logical forest plot.

    A forest plot with plot_appears_complete=False on one page is continued
    by the first forest plot on the next page (which shares the same title/headers).

    Returns a list of (title, DataFrame) tuples.
    """
    all_plots: list[tuple[str, list[str], list[list[str]]]] = []
    current_title = None
    current_headers = None
    current_rows: list[list[str]] = []

    for page_num in sorted(page_results.keys()):
        for fp in page_results[page_num].forest_plots:
            if current_title is not None and not fp.headers:
                # Continuation — append rows
                current_rows.extend(fp.rows)
            else:
                # New forest plot — save previous if exists
                if current_title is not None:
                    all_plots.append((current_title, current_headers, current_rows))
                current_title = fp.title
                current_headers = fp.headers
                current_rows = list(fp.rows)

    # Don't forget the last one
    if current_title is not None:
        all_plots.append((current_title, current_headers, current_rows))

    # Convert to DataFrames
    results = []
    for title, headers, rows in all_plots:
        df = pd.DataFrame(rows, columns=headers)
        results.append((title, df))

    return results
