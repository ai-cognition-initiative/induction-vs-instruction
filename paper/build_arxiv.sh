#!/usr/bin/env bash
# Package the arXiv version into a self-contained tarball for arXiv upload.
#
# Single-source: the canonical paper lives in paper/ as two thin wrappers
# (main.tex = COLM, arxiv.tex = arXiv) sharing paper/sections/ and paper/figures/.
# This script assembles the arXiv wrapper + shared sources into a flat bundle,
# copies outputs/plots/, rewrites \plotsroot to a local path, and pre-builds the
# .bbl so the upload is self-contained.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
plots_src="$repo_root/outputs/plots"
build="$here/arxiv_build"
out="$here/arxiv_submission.tar.gz"

[ -d "$plots_src" ] || { echo "missing $plots_src" >&2; exit 1; }

rm -rf "$build"
mkdir -p "$build"

# arXiv wrapper + shared sources
cp "$here/arxiv.tex" "$build/"
cp -R "$here/sections" "$build/sections"
cp -R "$here/figures" "$build/figures"

# Style + bibliography sources (arxiv.tex uses \bibliography{articles,colm2026_conference})
for f in arxiv.sty natbib.sty fancyhdr.sty colm2026_conference.bst articles.bib colm2026_conference.bib; do
  [ -f "$here/$f" ] && cp "$here/$f" "$build/"
done

cp -R "$plots_src" "$build/plots"

# Point \plotsroot at the bundled copy
sed -i 's|\\newcommand{\\plotsroot}{[^}]*}|\\newcommand{\\plotsroot}{plots/}|' "$build/arxiv.tex"

# Pre-build to generate arxiv.bbl so the upload does not depend on bibtex/.bib resolution
( cd "$build" \
  && pdflatex -interaction=nonstopmode arxiv.tex >/dev/null 2>&1 \
  && bibtex arxiv >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode arxiv.tex >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode arxiv.tex >/dev/null 2>&1 ) || \
  { echo "warning: pre-build failed; bundle still created but verify it compiles" >&2; }

# Tar everything except local build artifacts (keep arxiv.bbl)
( cd "$build" && tar -czf "$out" \
  --exclude='*.aux' --exclude='*.log' --exclude='*.out' --exclude='*.blg' \
  --exclude='*.fls' --exclude='*.fdb_latexmk' --exclude='*.synctex.gz' \
  . )
echo "wrote $out ($(du -h "$out" | cut -f1))"
