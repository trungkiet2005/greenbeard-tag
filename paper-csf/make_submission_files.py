"""Produce the separate editable files Chaos, Solitons & Fractals asks for at
submission: the highlights file and the declaration-of-competing-interest file.

Both must be uploaded as their own items in Editorial Manager, and the
declaration one must be .doc/.docx.

Run from paper-csf/:  python make_submission_files.py
"""

import re
from pathlib import Path

from docx import Document
from docx.shared import Pt

HERE = Path(__file__).resolve().parent

TITLE = ("Forgeable greenbeards: bistability and hollow collapse in the "
         "evolutionary dynamics of certified agent populations")

def _highlights_from_front_matter():
    r"""The highlights the typeset paper carries, read out of _front.tex.

    They used to be a second copy kept here by hand, and the two drifted: the
    manuscript said "Two disjoint attracting faces" where the uploaded .docx
    still said "The flow is bistable", and one bullet had lost its "at r=0".
    Editorial Manager takes the .docx and the reviewer reads the PDF, so the
    two must be the same sentence.  Reading one from the other is the only way
    that stays true.
    """
    front = (HERE / "_front.tex").read_text(encoding="utf-8")
    body = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}",
                     front, re.S)
    assert body, "no highlights environment in _front.tex"
    items = [line.strip()[len(r"\item"):].strip()
             for line in body.group(1).split("\n")
             if line.strip().startswith(r"\item")]
    assert items, "highlights environment is empty"
    return items


HIGHLIGHTS = _highlights_from_front_matter()

for h in HIGHLIGHTS:
    assert len(h) <= 85, f"highlight over 85 characters ({len(h)}): {h}"
assert 3 <= len(HIGHLIGHTS) <= 5, "CSF wants 3 to 5 highlights"


def new_doc():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    return doc


# ------------------------------------------------------------- highlights
doc = new_doc()
doc.add_heading("Highlights", level=1)
p = doc.add_paragraph()
p.add_run(TITLE).italic = True
for h in HIGHLIGHTS:
    doc.add_paragraph(h, style="List Bullet")
doc.save(HERE / "highlights.docx")

# --------------------------------------------- declaration of interests
doc = new_doc()
doc.add_heading("Declaration of competing interest", level=1)
p = doc.add_paragraph()
p.add_run("Manuscript title: ").bold = True
p.add_run(TITLE)
doc.add_paragraph(
    "The authors declare that they have no known competing financial interests "
    "or personal relationships that could have appeared to influence the work "
    "reported in this paper."
)
doc.add_heading("Funding", level=2)
doc.add_paragraph(
    "This research did not receive any specific grant from funding agencies in "
    "the public, commercial, or not-for-profit sectors."
)
doc.add_heading(
    "Declaration of generative AI and AI-assisted technologies in the "
    "manuscript preparation process",
    level=2,
)
doc.add_paragraph(
    "During the preparation of this work the authors used a large language "
    "model assistant in order to draft and edit prose, to scaffold parts of the "
    "analysis code, and to check the consistency of quoted numerical values "
    "against the generated results files. After using this tool the authors "
    "reviewed and edited the content as needed and take full responsibility for "
    "the content of the published article."
)
doc.save(HERE / "declaration_of_interest.docx")

print("wrote highlights.docx and declaration_of_interest.docx")
