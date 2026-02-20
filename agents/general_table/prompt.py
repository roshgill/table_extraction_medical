"""General table extraction prompt."""

GENERAL_TABLE_PROMPT = """Extract ALL tables from this scanned PDF page image.

INSTRUCTIONS:
- There may be multiple separate tables on this page — return each one as a separate entry
- For each table, identify the column headers and all data rows
- Return headers as a list of strings and each row as a list of cell values
- Preserve all text exactly as it appears (including leading zeros, punctuation, spacing)
- Each row must have the same number of values as there are headers
- If a cell is empty, return an empty string for that cell
- If a table continues beyond the bottom of this page, set table_appears_complete to false
- If a table is fully contained on this page, set table_appears_complete to true
- Return headers only if this page contains that table's header row
- Return tables in top-to-bottom order as they appear on the page
"""
