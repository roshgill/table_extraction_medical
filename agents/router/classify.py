"""Router agent — classifies table types found on a page."""

from PIL import Image

from .prompt import ROUTER_PROMPT


def classify_page(
    client,
    model: str,
    image: Image.Image,
) -> list[dict]:
    """Classify tables/figures found on a PDF page image.

    Args:
        client: google.genai.Client instance.
        model: Model name string.
        image: PIL image of the page.

    Returns:
        List of dicts with keys: figure_id, table_type, description.
    """
    raise NotImplementedError("Router agent not yet implemented")
