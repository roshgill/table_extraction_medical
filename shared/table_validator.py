"""
Table Extraction Validator
--------------------------
Structural quality checks on extracted HTML tables.
Produces per-table confidence scores and failure classifications.

Designed to sit between extraction (UniTable/Gemini) and downstream storage/RAG.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from html.parser import HTMLParser
from typing import Optional


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FailureType(Enum):
    COLUMN_MISMATCH = "column_mismatch"
    EMPTY_HEADER = "empty_header"
    MISSING_HEADER = "missing_header"
    EXCESSIVE_EMPTY_CELLS = "excessive_empty_cells"
    SPANNING_OVERFLOW = "spanning_overflow"
    DUPLICATE_ROWS = "duplicate_rows"
    OCR_ARTIFACTS = "ocr_artifacts"
    SINGLE_COLUMN = "single_column"
    WHITESPACE_DOMINANT = "whitespace_dominant"
    CELL_LENGTH_OUTLIER = "cell_length_outlier"
    ROW_COUNT_SUSPICIOUS = "row_count_suspicious"


@dataclass
class Issue:
    failure_type: FailureType
    severity: float  # 0-1, how much this drags confidence down
    detail: str


@dataclass
class ValidationResult:
    confidence: Confidence
    score: float  # 0-1
    issues: list = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    cell_count: int = 0
    empty_cell_count: int = 0
    has_header: bool = False

    @property
    def summary(self):
        tag = {
            Confidence.HIGH: "PASS",
            Confidence.MEDIUM: "REVIEW",
            Confidence.LOW: "FAIL",
        }[self.confidence]
        lines = [
            f"[{tag}] confidence={self.score:.2f}  "
            f"rows={self.row_count} cols={self.col_count} "
            f"cells={self.cell_count} empty={self.empty_cell_count} "
            f"header={'yes' if self.has_header else 'no'}"
        ]
        for iss in self.issues:
            lines.append(f"  - {iss.failure_type.value}: {iss.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML table parser -- builds a grid from raw HTML
# ---------------------------------------------------------------------------

class _TableParser(HTMLParser):
    """Parses an HTML table into a 2D grid, respecting colspan/rowspan."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: Optional[list] = None
        self._current_cell: Optional[list] = None
        self._spans: list[tuple[int, int, int]] = []  # (row, col, remaining_rowspan)
        self._cell_attrs: list[tuple[int, int]] = []  # (colspan, rowspan) per cell in current row
        self._in_thead = False
        self._header_rows = 0
        self._row_idx = 0

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if tag == "thead":
            self._in_thead = True
        elif tag in ("tr",):
            self._current_row = []
            self._cell_attrs = []
        elif tag in ("td", "th"):
            self._current_cell = []
            colspan = int(attrs_d.get("colspan", 1))
            rowspan = int(attrs_d.get("rowspan", 1))
            self._cell_attrs.append((colspan, rowspan))

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_thead = False
        elif tag in ("td", "th"):
            if self._current_cell is not None and self._current_row is not None:
                text = "".join(self._current_cell).strip()
                self._current_row.append(text)
            self._current_cell = None
        elif tag == "tr":
            if self._current_row is not None:
                self.rows.append(self._current_row)
                if self._in_thead:
                    self._header_rows += 1
                # store span info
                for cell_idx, (cs, rs) in enumerate(self._cell_attrs):
                    if rs > 1:
                        self._spans.append((self._row_idx, cell_idx, cs, rs))
                self._row_idx += 1
            self._current_row = None

    def handle_data(self, data):
        if self._current_cell is not None:
            self._current_cell.append(data)

    @property
    def has_thead(self):
        return self._header_rows > 0

    def effective_col_counts(self) -> list[int]:
        """Column count per row accounting for colspan (but not rowspan propagation)."""
        counts = []
        for row_idx, row in enumerate(self.rows):
            n = 0
            for cell_idx, _text in enumerate(row):
                if cell_idx < len(self._cell_attrs_by_row.get(row_idx, [])):
                    cs, _ = self._cell_attrs_by_row[row_idx][cell_idx]
                    n += cs
                else:
                    n += 1
            counts.append(n)
        return counts


class _TableParserV2(HTMLParser):
    """
    Simpler pass: just collect raw cell texts per row plus
    colspan/rowspan metadata. Good enough for validation heuristics.
    """

    def __init__(self):
        super().__init__()
        self.rows = []           # list of lists of cell text
        self.row_spans = []      # parallel: list of lists of (colspan, rowspan)
        self._cur_row = None
        self._cur_spans = None
        self._cur_cell = None
        self._cur_cs = 1
        self._cur_rs = 1
        self._in_thead = False
        self.header_row_count = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self._cur_row = []
            self._cur_spans = []
        elif tag in ("td", "th"):
            self._cur_cell = []
            self._cur_cs = int(a.get("colspan", 1))
            self._cur_rs = int(a.get("rowspan", 1))

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_thead = False
        elif tag in ("td", "th"):
            if self._cur_cell is not None and self._cur_row is not None:
                self._cur_row.append("".join(self._cur_cell).strip())
                self._cur_spans.append((self._cur_cs, self._cur_rs))
            self._cur_cell = None
        elif tag == "tr":
            if self._cur_row is not None:
                self.rows.append(self._cur_row)
                self.row_spans.append(self._cur_spans)
                if self._in_thead:
                    self.header_row_count += 1
            self._cur_row = None

    def handle_data(self, data):
        if self._cur_cell is not None:
            self._cur_cell.append(data)


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

# Common OCR garbage patterns
_OCR_JUNK = re.compile(
    r"[^\x00-\x7F\u00C0-\u024F\u2000-\u206F\u2190-\u21FF\u2200-\u22FF\u0370-\u03FF\u00B0-\u00BF%°±≤≥<>≈\u2264\u2265]"
)
_BROKEN_NUMBER = re.compile(r"\d+\s+\.\s+\d+")
# l/1 and O/0 swaps in contexts where they're clearly wrong
_L_FOR_1 = re.compile(r"(?<=[0-9])l|l(?=[0-9])")  # "l" adjacent to digits
_O_FOR_0 = re.compile(r"(?<=[0-9])O|O(?=[0-9])")  # "O" adjacent to digits  # "3 . 14" instead of "3.14"


def _effective_cols(row_cells, row_span_info):
    """Sum of colspans for a row."""
    total = 0
    for idx, _text in enumerate(row_cells):
        if idx < len(row_span_info):
            cs, _ = row_span_info[idx]
            total += cs
        else:
            total += 1
    return total


def validate_table(html: str) -> ValidationResult:
    """
    Run structural checks on an extracted HTML table.
    Returns a ValidationResult with confidence score and flagged issues.
    """
    if not html or not html.strip():
        return ValidationResult(
            confidence=Confidence.LOW,
            score=0.0,
            issues=[Issue(FailureType.MISSING_HEADER, 1.0, "empty input")],
        )

    parser = _TableParserV2()
    try:
        parser.feed(html)
    except Exception as e:
        return ValidationResult(
            confidence=Confidence.LOW,
            score=0.0,
            issues=[Issue(FailureType.OCR_ARTIFACTS, 1.0, f"HTML parse error: {e}")],
        )

    rows = parser.rows
    spans = parser.row_spans

    if not rows:
        return ValidationResult(
            confidence=Confidence.LOW,
            score=0.0,
            issues=[Issue(FailureType.MISSING_HEADER, 1.0, "no rows parsed")],
        )

    issues = []

    # --- basic counts ---
    all_cells = [cell for row in rows for cell in row]
    cell_count = len(all_cells)
    empty_count = sum(1 for c in all_cells if not c.strip())

    # effective column counts per row
    eff_cols = []
    for r_idx, (r, s) in enumerate(zip(rows, spans)):
        eff_cols.append(_effective_cols(r, s))

    # handle rowspan propagation (approximate):
    # build an "occupied" tracker
    occupied = {}  # (row, col) -> True
    adjusted_cols = list(eff_cols)
    for r_idx, (r, s) in enumerate(zip(rows, spans)):
        col_cursor = 0
        for c_idx, _text in enumerate(r):
            while (r_idx, col_cursor) in occupied:
                col_cursor += 1
            cs = s[c_idx][0] if c_idx < len(s) else 1
            rs = s[c_idx][1] if c_idx < len(s) else 1
            if rs > 1:
                for dr in range(1, rs):
                    for dc in range(cs):
                        occupied[(r_idx + dr, col_cursor + dc)] = True
                        # add to that row's effective count
                        if r_idx + dr < len(adjusted_cols):
                            adjusted_cols[r_idx + dr] += 1
            col_cursor += cs

    max_cols = max(adjusted_cols) if adjusted_cols else 0

    # --- Check 1: column consistency ---
    if max_cols > 0:
        mismatches = [i for i, c in enumerate(adjusted_cols) if c != max_cols]
        if mismatches:
            ratio = len(mismatches) / len(rows)
            sev = min(ratio * 2, 0.5)
            issues.append(Issue(
                FailureType.COLUMN_MISMATCH,
                sev,
                f"{len(mismatches)}/{len(rows)} rows deviate from expected {max_cols} cols"
            ))

    # --- Check 2: header presence ---
    has_header = parser.header_row_count > 0
    if not has_header:
        # check if first row looks like a header (all non-empty, shorter text)
        if rows:
            first_row_empty = sum(1 for c in rows[0] if not c.strip())
            if first_row_empty == 0 and len(rows) > 1:
                has_header = True  # probable header even without thead
            else:
                issues.append(Issue(
                    FailureType.MISSING_HEADER, 0.15,
                    "no <thead> and first row has empty cells"
                ))

    # header cells empty?
    if has_header and rows:
        header_cells = rows[0]
        empty_headers = sum(1 for c in header_cells if not c.strip())
        if empty_headers > 0 and len(header_cells) > 0:
            ratio = empty_headers / len(header_cells)
            if ratio > 0.3:
                issues.append(Issue(
                    FailureType.EMPTY_HEADER, ratio * 0.4,
                    f"{empty_headers}/{len(header_cells)} header cells are empty"
                ))

    # --- Check 3: empty cell ratio ---
    if cell_count > 0:
        empty_ratio = empty_count / cell_count
        if empty_ratio > 0.5:
            issues.append(Issue(
                FailureType.EXCESSIVE_EMPTY_CELLS, empty_ratio * 0.4,
                f"{empty_count}/{cell_count} cells empty ({empty_ratio:.0%})"
            ))

    # --- Check 4: spanning overflow ---
    for r_idx, s in enumerate(spans):
        for c_idx, (cs, rs) in enumerate(s):
            if cs > max_cols:
                issues.append(Issue(
                    FailureType.SPANNING_OVERFLOW, 0.3,
                    f"row {r_idx} cell {c_idx}: colspan={cs} exceeds table width {max_cols}"
                ))
            if rs > len(rows) - r_idx:
                issues.append(Issue(
                    FailureType.SPANNING_OVERFLOW, 0.2,
                    f"row {r_idx} cell {c_idx}: rowspan={rs} exceeds remaining rows"
                ))

    # --- Check 5: duplicate rows ---
    row_strs = ["|".join(r) for r in rows]
    seen = {}
    dup_count = 0
    for rs in row_strs:
        if rs in seen:
            dup_count += 1
        seen[rs] = True
    if dup_count > 0 and len(rows) > 2:
        issues.append(Issue(
            FailureType.DUPLICATE_ROWS,
            min(dup_count / len(rows), 0.4),
            f"{dup_count} duplicate rows"
        ))

    # --- Check 6: OCR artifacts ---
    junk_cells = 0
    broken_nums = 0
    for cell in all_cells:
        if _OCR_JUNK.search(cell):
            junk_cells += 1
        if _BROKEN_NUMBER.search(cell):
            broken_nums += 1
    if cell_count > 0:
        junk_ratio = junk_cells / cell_count
        if junk_ratio > 0.05:
            issues.append(Issue(
                FailureType.OCR_ARTIFACTS, junk_ratio * 0.5,
                f"{junk_cells} cells contain non-standard characters"
            ))
    if broken_nums > 0:
        issues.append(Issue(
            FailureType.OCR_ARTIFACTS,
            min(broken_nums * 0.05, 0.3),
            f"{broken_nums} cells have broken decimal numbers"
        ))

    # l/1 and O/0 substitution (very common in medical PDF OCR)
    subst_cells = 0
    for cell in all_cells:
        if _L_FOR_1.search(cell) or _O_FOR_0.search(cell):
            subst_cells += 1
    if subst_cells > 0:
        issues.append(Issue(
            FailureType.OCR_ARTIFACTS,
            min(subst_cells * 0.08, 0.35),
            f"{subst_cells} cells have likely l/1 or O/0 substitutions"
        ))

    # --- Check 7: single-column table ---
    if max_cols <= 1 and len(rows) > 1:
        issues.append(Issue(
            FailureType.SINGLE_COLUMN, 0.25,
            "table has only 1 column -- likely a parsing failure"
        ))

    # --- Check 8: whitespace-dominant cells ---
    ws_cells = 0
    for cell in all_cells:
        stripped = cell.strip()
        if stripped and len(stripped) < len(cell) * 0.3:
            ws_cells += 1
    if cell_count > 0 and ws_cells / cell_count > 0.2:
        issues.append(Issue(
            FailureType.WHITESPACE_DOMINANT,
            min(ws_cells / cell_count * 0.3, 0.2),
            f"{ws_cells} cells are mostly whitespace"
        ))

    # --- Check 9: cell length outliers ---
    if all_cells:
        lengths = [len(c) for c in all_cells if c.strip()]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            if avg_len > 0:
                outliers = sum(1 for l in lengths if l > avg_len * 10)
                if outliers > 0:
                    issues.append(Issue(
                        FailureType.CELL_LENGTH_OUTLIER, 0.15,
                        f"{outliers} cells are 10x+ longer than average ({avg_len:.0f} chars)"
                    ))

    # --- Check 10: suspicious row count ---
    if len(rows) > 200:
        issues.append(Issue(
            FailureType.ROW_COUNT_SUSPICIOUS, 0.1,
            f"{len(rows)} rows -- verify this isn't a multi-table merge"
        ))

    # --- compute final score ---
    penalty = sum(iss.severity for iss in issues)
    score = max(0.0, min(1.0, 1.0 - penalty))

    if score >= 0.8:
        conf = Confidence.HIGH
    elif score >= 0.5:
        conf = Confidence.MEDIUM
    else:
        conf = Confidence.LOW

    return ValidationResult(
        confidence=conf,
        score=round(score, 3),
        issues=issues,
        row_count=len(rows),
        col_count=max_cols,
        cell_count=cell_count,
        empty_cell_count=empty_count,
        has_header=has_header,
    )


def validate_against_ground_truth(
    predicted_html: str, ground_truth_html: str
) -> dict:
    """
    Compare predicted extraction against ground truth.
    Returns structural diff metrics useful for failure classification
    in the iterative rule-refinement loop.
    """
    pred_parser = _TableParserV2()
    gt_parser = _TableParserV2()
    pred_parser.feed(predicted_html)
    gt_parser.feed(ground_truth_html)

    pred_rows = pred_parser.rows
    gt_rows = gt_parser.rows

    result = {
        "row_count_match": len(pred_rows) == len(gt_rows),
        "pred_rows": len(pred_rows),
        "gt_rows": len(gt_rows),
        "row_diff": len(pred_rows) - len(gt_rows),
    }

    # column count comparison
    pred_cols = set()
    gt_cols = set()
    for r, s in zip(pred_rows, pred_parser.row_spans):
        pred_cols.add(_effective_cols(r, s))
    for r, s in zip(gt_rows, gt_parser.row_spans):
        gt_cols.add(_effective_cols(r, s))

    result["pred_col_counts"] = sorted(pred_cols)
    result["gt_col_counts"] = sorted(gt_cols)
    result["col_structure_match"] = pred_cols == gt_cols

    # cell-level text comparison (for aligned rows)
    matched_cells = 0
    total_cells = 0
    mismatched_examples = []
    min_rows = min(len(pred_rows), len(gt_rows))
    for r_idx in range(min_rows):
        min_cells = min(len(pred_rows[r_idx]), len(gt_rows[r_idx]))
        for c_idx in range(min_cells):
            total_cells += 1
            p = pred_rows[r_idx][c_idx].strip()
            g = gt_rows[r_idx][c_idx].strip()
            if p == g:
                matched_cells += 1
            elif len(mismatched_examples) < 5:
                mismatched_examples.append({
                    "row": r_idx, "col": c_idx,
                    "predicted": p[:80], "ground_truth": g[:80]
                })

    result["cell_accuracy"] = matched_cells / total_cells if total_cells > 0 else 0.0
    result["matched_cells"] = matched_cells
    result["total_compared"] = total_cells
    result["mismatch_examples"] = mismatched_examples

    # classify the dominant failure mode
    failure_modes = []
    if not result["row_count_match"]:
        failure_modes.append("row_count")
    if not result["col_structure_match"]:
        failure_modes.append("column_structure")
    if result["cell_accuracy"] < 0.9 and result["row_count_match"]:
        failure_modes.append("cell_content")  # structure ok, text wrong (OCR)
    if result["cell_accuracy"] >= 0.9 and result["row_count_match"]:
        failure_modes.append("minor_text_diff")

    result["failure_modes"] = failure_modes

    return result


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def validate_batch(extractions: dict[str, str]) -> dict[str, ValidationResult]:
    """
    Validate a dict of {filename: html_string}.
    Returns parallel dict of validation results.
    """
    return {fname: validate_table(html) for fname, html in extractions.items()}


def batch_summary(results: dict[str, ValidationResult]) -> str:
    """Print a compact summary of batch validation."""
    high = sum(1 for r in results.values() if r.confidence == Confidence.HIGH)
    med = sum(1 for r in results.values() if r.confidence == Confidence.MEDIUM)
    low = sum(1 for r in results.values() if r.confidence == Confidence.LOW)
    total = len(results)

    lines = [
        f"Batch: {total} tables validated",
        f"  HIGH confidence:   {high}/{total}",
        f"  MEDIUM (review):   {med}/{total}",
        f"  LOW (failed):      {low}/{total}",
        "",
    ]

    # show failures and reviews
    for fname, res in sorted(results.items(), key=lambda x: x[1].score):
        if res.confidence != Confidence.HIGH:
            lines.append(f"  {fname}")
            lines.append(f"    {res.summary}")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # a well-formed table
    good_html = """
    <html><body><table>
    <thead><tr><th>Drug</th><th>Dose (mg)</th><th>Response Rate</th></tr></thead>
    <tbody>
    <tr><td>Atorvastatin</td><td>10</td><td>45.2%</td></tr>
    <tr><td>Rosuvastatin</td><td>5</td><td>48.1%</td></tr>
    <tr><td>Simvastatin</td><td>20</td><td>38.7%</td></tr>
    </tbody>
    </table></body></html>
    """

    # a broken table (missing cells, no header, OCR junk)
    bad_html = """
    <html><body><table>
    <tr><td></td><td>Col A</td><td>Col B</td><td>Col C</td></tr>
    <tr><td>Row 1</td><td>12.3</td></tr>
    <tr><td>Row 2</td><td></td><td></td><td></td></tr>
    <tr><td>Row 2</td><td></td><td></td><td></td></tr>
    <tr><td>Row 3</td><td>4 . 56</td><td></td><td>val</td></tr>
    </table></body></html>
    """

    print("=== Good table ===")
    r = validate_table(good_html)
    print(r.summary)

    print("\n=== Broken table ===")
    r = validate_table(bad_html)
    print(r.summary)

    print("\n=== Ground truth comparison ===")
    diff = validate_against_ground_truth(bad_html, good_html)
    for k, v in diff.items():
        print(f"  {k}: {v}")