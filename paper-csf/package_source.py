"""Build the deterministic CSF LaTeX source archive from an explicit allowlist.

Run after ``assemble.py`` and a successful LaTeX/BibTeX build.  The allowlist
is deliberate: private submission notes, rendered manuscript PDFs, auxiliary
files, and the retired graphical-abstract artifacts cannot enter the archive by
accident.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "CSF_manuscript_source.zip"
ROOT_FILES = (
    "main.tex",
    "main.bbl",
    "refs.bib",
    "tables_generated.tex",
    "robustness_generated.tex",
    "cas-sc.cls",
    "cas-common.sty",
)
FIGURES = tuple(f"figures/fig{i:02d}_{name}.pdf" for i, name in enumerate(
    (
        "exclusion", "handshake", "invasion", "cliff", "nucleation",
        "instruments", "channels", "providers", "robustness",
    ),
    start=1,
))
THUMBNAILS = tuple(
    f"thumbnails/cas-{name}.{suffix}"
    for name, suffix in (
        ("email", "jpeg"), ("facebook", "jpeg"), ("gplus", "jpeg"),
        ("linkedin", "jpeg"), ("twitter", "jpeg"), ("url", "jpeg"),
    )
)
MEMBERS = ROOT_FILES + FIGURES + THUMBNAILS


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def main() -> None:
    missing = [name for name in MEMBERS if not (HERE / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source archive inputs missing: {missing}")
    if any("graphical" in name.lower() for name in MEMBERS):
        raise RuntimeError("retired graphical abstract entered source allowlist")

    payloads = {name: (HERE / name).read_bytes() for name in MEMBERS}
    manifest = "".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n"
        for name in MEMBERS
    ).encode("ascii")

    with ZipFile(ARCHIVE, "w") as archive:
        for name in MEMBERS:
            archive.writestr(_info(name), payloads[name])
        archive.writestr(_info("SOURCE_MANIFEST.sha256"), manifest)

    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    print(f"wrote {ARCHIVE.name}: {len(MEMBERS)} files + manifest")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
