#!/usr/bin/env python3
"""Reformat docs/acceptance-register.md's tables into per-row blocks + index.

This is a formatting-only pass (doc-restructure T6b): each `| D1 | ... |`
table row in sections A/B/P becomes its own `### <ID> <title>` block with
four labelled paragraphs (Issue/Detail/Recommendation/Disposition), and a
new index list is inserted at the top (one line per row: ID, a coarse
status pulled from the row's own Disposition text, and an 8-word gist).
Everything that is not a table row -- the file's title/metadata preamble,
each section's own intro prose, section C (no table), and the trailing
attribution lines -- is copied through unchanged.

CONTENT-IDENTITY PROOF: `dump_cells()` walks the table rows (before) and the
generated blocks (after) and returns the same flat list of cell text in the
same order, whitespace-collapsed. `diff -u` of the two dumps must be empty;
this script computes and prints that diff and refuses to write the
reformatted file if it is not empty.

Usage:
    dev/bin/reformat-register.py [--check] [path/to/acceptance-register.md]

`--check` verifies the identity proof and prints the diff without writing
the reformatted content back to disk.
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "docs" / "acceptance-register.md"

SECTION_A = "## A. Behavioral deviations"
SECTION_B = "## B. Possible full-parity exceptions"
SECTION_P = "## P. Performance backlog"
SECTION_C = "## C. Required later work, not drop candidates"
TABLE_SECTIONS = [SECTION_A, SECTION_B, SECTION_P]

ROW_ID_RE = re.compile(r"^\| ([DMP]\d+[a-z]?) \|")

# Process-jargon tokens that must never appear in NEW prose this script
# authors (index gists, derived block titles). Reproduced cell text (the
# Issue/Detail/Recommendation/Disposition paragraph bodies) is left alone --
# it is copied verbatim and governed by the content-identity proof, not this
# rule.
JARGON_RE = re.compile(
    r"\b(SR1|Task-?\d+|bucket|wave\d*|P\d+(?:\.\d+)?(?:-[A-Za-z]+)?)\b"
)

STATUS_RE = re.compile(
    r"\b(DROP|RESTORE|DEFER|ACCEPT|IMPLEMENTED|FIXED|ADOPTED|DISCHARGED|"
    r"INVESTIGATED|approved)\b",
    re.IGNORECASE,
)
# The 2026-07-28 SR1 S5b batch ratification wrote every row it touched in
# this exact form; where present, the verb right after it is the row's
# authoritative, most-recent verdict and takes priority over any other
# status word appearing later in the same cell (e.g. a parenthetical aside
# like D31's "DEFER (partially discharged)").
BATCH_RE = re.compile(r"ratified 2026-07-28 \(owner, batch\):\s*([A-Z]+)")
# D24/D26 each carry a dated follow-up that explicitly overturns the batch
# verdict ("owner reversed DEFER"); once that phrase appears, the first
# status-shaped word found after it (a slightly wider vocabulary than
# STATUS_RE -- "ported"/"promoted" also count here) is the current verdict.
REVERSAL_RE = re.compile(r"owner reversed \w+", re.IGNORECASE)
REVERSAL_TAIL_RE = re.compile(
    r"\b(DROP|RESTORE|DEFER|ACCEPT|IMPLEMENTED|FIXED|ADOPTED|DISCHARGED|"
    r"INVESTIGATED|PORTED|PROMOTED|approved)\b",
    re.IGNORECASE,
)


def split_row(line):
    """Split one `| a | b | c |` table row on unescaped `|`, respecting
    backtick code spans (a cell may legitimately contain a literal `|`
    inside backticks, e.g. `` `|H_ex|=0` `` in D19)."""
    body = line.strip()
    assert body.startswith("|") and body.endswith("|"), line
    body = body[1:-1]
    cells, cur, in_code = [], [], False
    for ch in body:
        if ch == "`":
            in_code = not in_code
            cur.append(ch)
        elif ch == "|" and not in_code:
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur))
    return [c.strip() for c in cells]


def normalize(text):
    """Whitespace-collapsed form used by the content-identity dump."""
    return re.sub(r"\s+", " ", text).strip()


def degargonize(text):
    """Strip process-jargon tokens from DERIVED prose only (titles/gists).
    Drops a parenthetical aside entirely if it contains a jargon token,
    then removes any remaining bare jargon token."""

    def strip_parens(m):
        return "" if JARGON_RE.search(m.group(0)) else m.group(0)

    text = re.sub(r"\([^()]*\)", strip_parens, text)
    text = JARGON_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" ,;-")


def find_first_clause_end(text):
    """Position of the first sentence-ish break (". "/"; "/" -- "/" — ")
    OUTSIDE any backtick code span -- so an inline-code dot, like the one in
    `` `Simulation.save_m_in_region` ``, is never mistaken for a clause
    break. Returns None if no such break exists."""
    in_code = False
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == "`":
            in_code = not in_code
        elif not in_code:
            if ch in ".;" and (i + 1 == n or text[i + 1].isspace()):
                return i
            if text[i : i + 4] == " -- " or text[i : i + 3] == " — ":
                return i
        i += 1
    return None


def close_dangling_markup(s):
    """If truncation left an odd number of backticks, drop back to before
    the last opening backtick rather than emit an unterminated code span."""
    if s.count("`") % 2 == 1:
        s = s[: s.rfind("`")].rstrip()
    return s


def strip_strikethrough_markers(text):
    """Titles/gists are short derived summaries, not a markdown-fidelity
    copy -- drop `~~` strikethrough delimiters outright (keeping the
    enclosed text) rather than track their balance through truncation.
    D26 is the register's one `~~...~~` row."""
    return text.replace("~~", "")


def make_title(row_id, col2):
    text = strip_strikethrough_markers(degargonize(col2))
    end = find_first_clause_end(text)
    first = text[:end] if end is not None else text
    if len(first) > 90:
        first = close_dangling_markup(first[:90].rsplit(" ", 1)[0]) + "…"
    first = first.strip(" ,")
    return f"{row_id} {first}" if first else row_id


def make_gist(col2):
    text = strip_strikethrough_markers(degargonize(col2))
    words = text.split()
    gist = " ".join(words[:8])
    if len(words) > 8:
        gist = close_dangling_markup(gist) + "…"
    return gist


def make_status(disposition):
    reversal = REVERSAL_RE.search(disposition)
    if reversal:
        m = REVERSAL_TAIL_RE.search(disposition, reversal.end())
        if m:
            return m.group(1).upper()
    batch = BATCH_RE.search(disposition)
    if batch:
        return batch.group(1).upper()
    matches = list(STATUS_RE.finditer(disposition))
    if matches:
        return matches[-1].group(1).upper()
    if "documented" in disposition.lower():
        return "NOTED"
    # The register's own preamble states no row remains open awaiting a
    # decision; a disposition cell with no matched keyword still records a
    # finalised owner call, just not one of the standard verbs above.
    return "RESOLVED"


def extract_section(lines, start, end):
    """start/end = indices of this section's '## ...' heading and the next
    one (end exclusive). Returns (heading, intro_lines, rows)."""
    sec = lines[start:end]
    heading = sec[0]
    hdr_idx = next(k for k, l in enumerate(sec) if l.startswith("| ID |"))
    intro = sec[1:hdr_idx]
    sep_idx = hdr_idx + 1
    assert sec[sep_idx].startswith("|---"), sec[sep_idx]
    row_lines = sec[sep_idx + 1 :]
    while row_lines and row_lines[-1].strip() == "":
        row_lines = row_lines[:-1]
    rows = []
    for rl in row_lines:
        m = ROW_ID_RE.match(rl)
        if not m:
            raise ValueError(f"unexpected non-row line in {heading!r}: {rl!r}")
        cells = split_row(rl)
        if len(cells) != 5:
            raise ValueError(f"row {cells[0] if cells else '?'} has {len(cells)} cells, expected 5")
        rows.append(
            {"id": cells[0], "col2": cells[1], "col3": cells[2], "col4": cells[3], "col5": cells[4]}
        )
    return heading, intro, rows


def parse_original(lines):
    idx = {h: lines.index(h) for h in (SECTION_A, SECTION_B, SECTION_P, SECTION_C)}
    preamble = lines[: idx[SECTION_A]]
    tail = lines[idx[SECTION_C] :]
    bounds = [
        (SECTION_A, idx[SECTION_A], idx[SECTION_B]),
        (SECTION_B, idx[SECTION_B], idx[SECTION_P]),
        (SECTION_P, idx[SECTION_P], idx[SECTION_C]),
    ]
    sections = [extract_section(lines, s, e) for _, s, e in bounds]
    return preamble, sections, tail


def make_block(row):
    return [
        f"### {make_title(row['id'], row['col2'])}",
        "",
        f"**Issue:** {row['col2']}",
        "",
        f"**Detail:** {row['col3']}",
        "",
        f"**Recommendation:** {row['col4']}",
        "",
        f"**Disposition:** {row['col5']}",
        "",
    ]


def build_index(all_rows):
    out = ["## Index", ""]
    for r in all_rows:
        out.append(f"- {r['id']} — {make_status(r['col5'])} — {make_gist(r['col2'])}")
    out.append("")
    return out


def reformat(text):
    lines = text.splitlines()
    preamble, sections, tail = parse_original(lines)
    all_rows = [r for (_, _, rows) in sections for r in rows]

    new_lines = list(preamble)
    new_lines.extend(build_index(all_rows))
    for heading, intro, rows in sections:
        new_lines.append(heading)
        new_lines.extend(intro)
        for r in rows:
            new_lines.extend(make_block(r))
    new_lines.extend(tail)
    return "\n".join(new_lines).rstrip("\n") + "\n", len(all_rows)


def dump_cells_original(text):
    lines = text.splitlines()
    _, sections, _ = parse_original(lines)
    cells = []
    for _, _, rows in sections:
        for r in rows:
            cells += [r["id"], r["col2"], r["col3"], r["col4"], r["col5"]]
    return [normalize(c) for c in cells]


def get_field(text, label, next_label):
    marker = f"**{label}:** "
    start = text.index(marker) + len(marker)
    end = text.index(f"**{next_label}:**", start) if next_label else len(text)
    return text[start:end]


def dump_cells_reformatted(text):
    lines = text.splitlines()
    cells = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith("### "):
            row_id = lines[i][4:].split(" ", 1)[0]
            j = i + 1
            block = []
            while j < n and not lines[j].startswith("### ") and not lines[j].startswith("## "):
                block.append(lines[j])
                j += 1
            body = "\n".join(block)
            issue = get_field(body, "Issue", "Detail")
            detail = get_field(body, "Detail", "Recommendation")
            rec = get_field(body, "Recommendation", "Disposition")
            disp = get_field(body, "Disposition", None)
            cells += [row_id, issue, detail, rec, disp]
            i = j
        else:
            i += 1
    return [normalize(c) for c in cells]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default=str(DEFAULT_PATH))
    ap.add_argument(
        "--check", action="store_true", help="verify the identity proof; do not write"
    )
    args = ap.parse_args()
    path = Path(args.path)
    original_text = path.read_text()

    new_text, n_rows = reformat(original_text)

    before = dump_cells_original(original_text)
    after = dump_cells_reformatted(new_text)

    diff = list(
        difflib.unified_diff(
            before, after, fromfile="before-dump", tofile="after-dump", lineterm=""
        )
    )
    print(f"Rows converted: {n_rows}")
    print(f"Content-identity dump: before={len(before)} cells, after={len(after)} cells")
    if diff:
        print("CONTENT-IDENTITY PROOF FAILED -- diff is non-empty:")
        for l in diff:
            print(l)
        return 1
    print("CONTENT-IDENTITY PROOF: diff is EMPTY (before-dump == after-dump).")

    if args.check:
        print("--check: not writing (dry run).")
        return 0

    path.write_text(new_text)
    print(f"Wrote reformatted register to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
