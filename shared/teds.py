"""
teds.py
-------
Tree Edit Distance based Similarity (TEDS) — the official PubTabNet metric.

TEDs = 1 - EditDistance(T_pred, T_gt) / max(|T_pred|, |T_gt|)

Where T is the parse tree of the table HTML and |T| is the number of nodes.
Score is in [0, 1], higher is better. 1.0 = perfect structural match.

Two modes:
  teds(pred, gt)            — full TEDs (structure + cell content)
  teds(pred, gt, structure_only=True) — TEDs-S (structure only, ignore cell text)

Reference: Zhong et al. "Image-based table recognition: data, model, and evaluation"
           ECCV 2020 — https://arxiv.org/abs/1911.10683
"""

from bs4 import BeautifulSoup, NavigableString
from apted import APTED, Config
from apted.helpers import Tree
from dataclasses import dataclass
from typing import Optional
import re


# ── tree node ──────────────────────────────────────────────────────────────────

@dataclass
class TableNode:
    tag:      str          # 'table', 'thead', 'tbody', 'tr', 'td', 'th', or '#text'
    content:  str = ""     # cell text (only for leaf td/th nodes in full mode)

    def __str__(self):
        # apted uses str(node) as the label for edit distance
        return f"{self.tag}:{self.content}" if self.content else self.tag


class TableTree:
    """Wrapper so apted can walk our TableNode tree."""
    def __init__(self, node: TableNode, children: list = None):
        self.node     = node
        self.children = children or []

    def __str__(self):
        return str(self.node)


class TEDSConfig(Config):
    """Tell apted how to compare nodes and get children."""

    def rename(self, node1, node2) -> float:
        """Cost of relabeling node1 → node2. 0 = same, 1 = different."""
        return 0 if str(node1) == str(node2) else 1

    def children(self, node):
        return node.children


# ── HTML → tree ────────────────────────────────────────────────────────────────

_IGNORED_TAGS = {"br", "b", "i", "u", "em", "strong", "span", "sup", "sub",
                 "font", "small", "p", "a"}


def _normalize_text(text: str) -> str:
    """Collapse whitespace, strip, lowercase for comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _soup_to_tree(element, structure_only: bool = False) -> Optional[TableTree]:
    """Recursively convert a BeautifulSoup element to a TableTree."""

    if isinstance(element, NavigableString):
        text = str(element).strip()
        if not text:
            return None
        if structure_only:
            return None   # ignore text nodes in structure-only mode
        return TableTree(TableNode("#text", _normalize_text(text)))

    tag = element.name.lower() if element.name else ""

    if tag in _IGNORED_TAGS:
        # flatten — treat children as if they belong to the parent
        children = []
        for child in element.children:
            t = _soup_to_tree(child, structure_only)
            if t:
                children.append(t)
        if not children:
            return None
        if len(children) == 1:
            return children[0]
        # wrap in a synthetic node to preserve structure
        node = TableTree(TableNode("inline"), children)
        return node

    if tag not in ("table", "thead", "tbody", "tfoot", "tr", "td", "th"):
        return None

    children = []
    for child in element.children:
        t = _soup_to_tree(child, structure_only)
        if t:
            children.append(t)

    # for td/th in full mode: attach cell text as content (not as child text nodes)
    content = ""
    if not structure_only and tag in ("td", "th"):
        raw = element.get_text(" ", strip=True)
        content = _normalize_text(raw)
        # remove text children — content is already captured above
        children = [c for c in children if c.node.tag != "#text"]

    return TableTree(TableNode(tag, content), children)


def html_to_tree(html: str, structure_only: bool = False) -> Optional[TableTree]:
    """Parse an HTML string and return the table parse tree."""
    soup  = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        return None
    return _soup_to_tree(table, structure_only)


def _count_nodes(tree: TableTree) -> int:
    if tree is None:
        return 0
    return 1 + sum(_count_nodes(c) for c in tree.children)


# ── TEDs score ─────────────────────────────────────────────────────────────────

def teds(pred_html: str, gt_html: str, structure_only: bool = False) -> float:
    """
    Compute TEDs between predicted and ground-truth table HTML.

    Returns float in [0, 1]. Returns 0.0 if either tree is unparseable.
    """
    t_pred = html_to_tree(pred_html, structure_only)
    t_gt   = html_to_tree(gt_html,   structure_only)

    if t_pred is None or t_gt is None:
        return 0.0

    n_pred = _count_nodes(t_pred)
    n_gt   = _count_nodes(t_gt)
    denom  = max(n_pred, n_gt)

    if denom == 0:
        return 1.0

    edit_dist = APTED(t_pred, t_gt, TEDSConfig()).compute_edit_distance()
    score     = 1.0 - edit_dist / denom
    return max(0.0, round(score, 4))


def teds_batch(pred_gt_pairs: list, structure_only: bool = False) -> dict:
    """
    Compute TEDs for a list of (pred_html, gt_html) pairs.
    Returns {'mean': float, 'scores': [float], 'perfect': int, 'poor': int}
    """
    scores = [teds(p, g, structure_only) for p, g in pred_gt_pairs]
    return {
        "mean":     round(sum(scores) / len(scores), 4) if scores else 0.0,
        "scores":   scores,
        "perfect":  sum(1 for s in scores if s >= 0.99),
        "good":     sum(1 for s in scores if 0.80 <= s < 0.99),
        "poor":     sum(1 for s in scores if s < 0.50),
        "n":        len(scores),
    }


# ── quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gt = """
    <table>
      <thead><tr><th>Drug</th><th>Dose (mg)</th><th>Response (%)</th></tr></thead>
      <tbody>
        <tr><td>Atorvastatin</td><td>10</td><td>38.2</td></tr>
        <tr><td>Rosuvastatin</td><td>5</td><td>42.1</td></tr>
      </tbody>
    </table>"""

    perfect = gt  # identical

    missing_col = """
    <table>
      <thead><tr><th>Drug</th><th>Dose (mg)</th></tr></thead>
      <tbody>
        <tr><td>Atorvastatin</td><td>10</td></tr>
        <tr><td>Rosuvastatin</td><td>5</td></tr>
      </tbody>
    </table>"""

    garbled = """
    <table>
      <tbody>
        <tr><td>Atorvastatin 10 38.2 Rosuvastatin 5 42.1</td></tr>
      </tbody>
    </table>"""

    print(f"Perfect match:     {teds(perfect, gt):.4f}  (expect ~1.0)")
    print(f"Missing column:    {teds(missing_col, gt):.4f}  (expect ~0.7)")
    print(f"Collapsed table:   {teds(garbled, gt):.4f}  (expect ~0.3)")
    print(f"Structure-only:    {teds(missing_col, gt, structure_only=True):.4f}")