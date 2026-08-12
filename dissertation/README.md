# Dissertation LaTeX Workspace

This directory contains the local LaTeX dissertation workflow.

## Quick start
- Open the repository in VS Code.
- Install **LaTeX Workshop**.
- Open `dissertation/main.tex`.
- Build with LaTeX Workshop recipe (**latexmk (biber)**).

## Build from terminal
```bash
cd dissertation
latexmk -pdf main.tex
```

## Clean build artifacts
```bash
cd dissertation
latexmk -C
```

## Output
Compiled PDF and auxiliary files are written to:

- `dissertation/build/`

## Structure
- `chapters/` numbered chapter bodies
- `appendices/` appendix content
- `figures/` source figures (prefer PDF vector plots)
- `tables/` table assets
- `references.bib` bibliography database
