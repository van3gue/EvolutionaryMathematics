# Paper: Evolving step functions for the Erdős minimum overlap problem

Preliminary study. Everything needed to compile is in this folder:

```
paper/
  main.tex                       the paper
  figures/cost_curves.png        Figure 1  (regenerate: examples/erdos_continuous/plot_cost_curves.py)
  figures/overlap_visualization.png   Figure 2  (examples/erdos_continuous/visualize.py)
```

## Build

Any standard LaTeX install (no BibTeX needed; the bibliography is inline):

```bash
cd paper
pdflatex main.tex
pdflatex main.tex   # second pass resolves cross-references
```

Or upload this folder to Overleaf and compile `main.tex`.

Packages used: `geometry, amsmath, amssymb, amsthm, graphicx, booktabs, hyperref, caption`.

## Before sharing

- Fill in `\author{...}` in `main.tex`.
- The four bibliography entries were written from memory and **have not been
  verified** against the sources. Check them.
- The numbers are a snapshot of runs cut short by API-credit exhaustion
  (see Section 5 of the paper). `.tex` has not been test-compiled on the
  machine that wrote it; static checks only (balanced environments, figures
  present, no undefined references/citations).
