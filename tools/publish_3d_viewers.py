#!/usr/bin/env python3
"""Prepare two local 3D-IMC HTML viewers for GitHub Pages without modifying raw research files.

Dry run by default. --apply copies *verified self-contained or CDN-backed* HTML to
an existing GitHub export repo and updates its README. Does not push to GitHub.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import sys

VIEWERS = {
    "roi_voxel_viewer.html": ("3D ROI voxel viewer", "Explore a reconstructed 3D tissue region at voxel level."),
    "centroid_marker_viewer.html": ("3D centroid and marker viewer", "Explore spatial pseudo-instance centroids and protein-marker measurements."),
}
HTML_RESOURCE_TAGS = {
    "script": "src", "img": "src", "iframe": "src", "link": "href",
    "source": "src", "video": "src", "audio": "src", "object": "data",
}
LOCAL_PATH_PATTERN = re.compile(r"(?:file:(?://)?|/(?:media|home|mnt|Users)/|\\\\(?:Users|home)\\\\)", re.I)
FETCH_PATTERN = re.compile(r"(?:fetch|open)\s*\(\s*['\"]([^'\"]+)['\"]", re.I)
CSS_URL_PATTERN = re.compile(r"url\(\s*['\"]?([^)'\"]+)", re.I)


class ResourceScanner(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.resources: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        d = dict(attrs)
        attr = HTML_RESOURCE_TAGS.get(tag)
        if attr and d.get(attr):
            self.resources.append(d[attr])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def assess(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        return [f"Missing: {path}"], []
    size = path.stat().st_size
    if size >= 95 * 1024 * 1024:
        errors.append(f"HTML exceeds safe GitHub file threshold (95 MiB): {path.name} ({size / 1048576:.1f} MiB)")
    elif size > 20 * 1024 * 1024:
        warnings.append(f"Large HTML may load slowly: {path.name} ({size / 1048576:.1f} MiB)")
    raw = path.read_text(encoding="utf-8", errors="replace")
    if "<html" not in raw.lower():
        warnings.append(f"No <html> tag detected: {path.name}")
    if LOCAL_PATH_PATTERN.search(raw) or re.search(r"(?:localhost|127\.0\.0\.1)(?::\d+)?", raw, re.I):
        errors.append(f"Potential local file/address reference detected in {path.name}; inspect before publishing")
    scan = ResourceScanner()
    scan.feed(raw)
    links = scan.resources + CSS_URL_PATTERN.findall(raw) + FETCH_PATTERN.findall(raw)
    for dep in sorted(set(links)):
        dep = dep.strip()
        if dep.startswith(("https://", "http://", "data:", "#", "blob:")):
            if dep.startswith(("http://", "https://")):
                warnings.append(f"External resource (requires network availability): {dep[:120]}")
            continue
        if dep.startswith("//"):
            warnings.append(f"Protocol-relative remote dependency: {dep[:120]}")
            continue
        errors.append(f"Nonportable resource reference in {path.name}: {dep[:140]}")
    return errors, sorted(set(warnings))


def landing_html() -> str:
    cards = "\n".join(
        f'<a class="card" href="{name}"><span class="label">Interactive viewer</span>'
        f'<h2>{html.escape(title)}</h2><p>{html.escape(description)}</p>'
        f'<span class="open">Open 3D viewer &rarr;</span></a>'
        for name, (title, description) in VIEWERS.items()
    )
    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D-TissueMap AI | Interactive 3D viewers</title>
<style>body{{font-family:system-ui,-apple-system,sans-serif;max-width:980px;margin:5vh auto;padding:0 22px;color:#243247;line-height:1.6;background:#f8fafc}}
h1{{font-size:clamp(28px,4vw,44px)}}.small{{color:#5a687a}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:22px;margin:30px 0}}
.card{{display:block;text-decoration:none;color:inherit;padding:25px;border-radius:16px;background:white;border:1px solid #dce2e9;box-shadow:0 4px 14px #1232  }}
.card:hover{{border-color:#2583a8}}.label{{font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:#397e92}}.open{{color:#087fa2;font-weight:650}}
.note{{font-size:14px;color:#627083;border-left:3px solid #a6b7c4;padding-left:14px}}</style>
</head><body><p class="small">3D-TissueMap AI / Visualization</p><h1>Interactive 3D tissue viewers</h1>
<p>Explore research visualizations derived from imaging mass cytometry. These displays use previously reconstructed tissue data and pseudo-instance representations.</p>
<div class="grid">{cards}</div>
<p class="note">Research preview, not a validated cell-level ground truth. Data provenance, marker identity and redistribution rights should be checked before public release.</p>
</body></html>'''


def readme_section(url: str) -> str:
    return f'''<!-- 3D_VIEWERS_START -->
## Interactive 3D tissue viewers

Explore the reconstructed imaging mass cytometry tissue using two browser-based viewers. These are **research visualizations**, not manual ground-truth cell segmentations.

| Interactive tool | Description | Launch |
|---|---|---|
| 3D ROI voxel viewer | Voxel-based 3D tissue visualization | [Open viewer]({url}/roi_voxel_viewer.html) |
| 3D centroid and marker viewer | 3D spatial pseudo-instances and protein-marker visualization | [Open viewer]({url}/centroid_marker_viewer.html) |

**[Open the 3D viewer gallery]({url}/index.html)** · [Viewer documentation](docs/VIEWERS.md)

> The interactive HTML files are distributed as static pages. The underlying 3D pseudo-instance geometry does not constitute independently validated cell ground truth. Marker labels and redistribution rights require verification.
<!-- 3D_VIEWERS_END -->'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd(), help="Original local 3D-TissueMap-AI root")
    p.add_argument("--export-repo", type=Path, default=None, help="GitHub export directory (default: sibling 3D-TissueMap-AI-GitHub)")
    p.add_argument("--owner", default="Great-sophie")
    p.add_argument("--repo-name", default="3D-TissueMap-AI")
    p.add_argument("--apply", action="store_true", help="Copy viewers and update export README (dry run by default)")
    args = p.parse_args()

    root = args.project_root.expanduser().resolve()
    export = (args.export_repo or root.parent / "3D-TissueMap-AI-GitHub").expanduser().resolve()
    srcdir = root / "runs" / "3d_tissue_microenvironment_viewer_v1"
    dstdir = export / "docs" / "viewers"
    url = f"https://{args.owner.lower()}.github.io/{args.repo_name}/viewers"
    print(f"Original project: {root}")
    print(f"GitHub export:     {export}")
    print(f"Source:            {srcdir}")
    print(f"Pages expected:    {url}/index.html")

    if not export.is_dir() or not (export / "README.md").is_file():
        sys.exit("STOP: GitHub export and its README.md must already exist. Run the GitHub preparation step first.")

    issues, warnings = [], []
    for fname in VIEWERS:
        src = srcdir / fname
        errors, cautions = assess(src)
        issues += errors
        warnings += cautions
        if src.is_file():
            print(f"  {fname}: {src.stat().st_size / 1048576:.2f} MiB, sha256={sha256(src)[:16]}...")
    for w in sorted(set(warnings))[:15]:
        print("WARN:", w)
    if len(warnings) > 15:
        print(f"WARN: {len(warnings) - 15} additional external/dependency notices omitted")
    for i in issues:
        print("BLOCKER:", i)
    if issues:
        sys.exit("STOP: fix local dependencies, private references or missing files before publishing.")

    for fname in VIEWERS:
        src, dest = srcdir / fname, dstdir / fname
        if dest.is_file() and sha256(dest) != sha256(src):
            sys.exit(f"STOP: existing viewer differs: {dest}. Back it up/review it before replacing it.")

    if not args.apply:
        print("DRY RUN PASS. No files modified. Run again with --apply to publish into the local export.")
        return

    dstdir.mkdir(parents=True, exist_ok=True)
    (export / "manifests").mkdir(parents=True, exist_ok=True)
    (export / "docs").mkdir(parents=True, exist_ok=True)
    for name in VIEWERS:
        src, dest = srcdir / name, dstdir / name
        if not dest.exists():
            shutil.copy2(src, dest)
            print(f"COPIED {dest.relative_to(export)}")
        else:
            print(f"UNCHANGED {dest.relative_to(export)}")
    (dstdir / "index.html").write_text(landing_html(), encoding="utf-8")
    (export / "docs" / "VIEWERS.md").write_text(
        "# Interactive 3D viewers\n\n"
        "The two HTML viewers originate from `runs/3d_tissue_microenvironment_viewer_v1/`. "
        "They are copied to `docs/viewers/` for GitHub Pages. They contain research visualization data "
        "and must be reviewed for public redistribution before pushing.\n\n"
        f"- [ROI voxel viewer]({url}/roi_voxel_viewer.html)\n"
        f"- [Centroid marker viewer]({url}/centroid_marker_viewer.html)\n"
        f"- [Viewer landing page]({url}/index.html)\n\n"
        "For a local preview, run `python -m http.server 8000 --directory docs` from the GitHub export root, "
        "then open `http://localhost:8000/viewers/`.\n",
        encoding="utf-8",
    )
    readme_path = export / "README.md"
    original = readme_path.read_text(encoding="utf-8")
    marker_start, marker_end = "<!-- 3D_VIEWERS_START -->", "<!-- 3D_VIEWERS_END -->"
    block = readme_section(url)
    if marker_start in original and marker_end in original:
        a = original.index(marker_start)
        b = original.index(marker_end, a) + len(marker_end)
        changed = original[:a] + block + original[b:]
    elif marker_start not in original and marker_end not in original:
        changed = original.rstrip() + "\n\n" + block + "\n"
    else:
        sys.exit("STOP: README contains only one 3D viewer marker; resolve manually")
    readme_path.write_text(changed, encoding="utf-8")
    manifest = export / "manifests" / "viewers_sha256.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo_relative_path", "sha256", "bytes"])
        for fname in VIEWERS:
            fp = dstdir / fname
            w.writerow([fp.relative_to(export).as_posix(), sha256(fp), fp.stat().st_size])
    print("SUCCESS: both viewers, gallery, README section, provenance and checksums prepared.")
    print("NEXT: run your prepush audit, inspect staged diff, commit, push; then enable GitHub Pages from main /docs.")


if __name__ == "__main__":
    main()
