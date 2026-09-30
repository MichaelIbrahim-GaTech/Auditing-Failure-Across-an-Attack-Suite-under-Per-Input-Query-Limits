# Auditing Failure Across an Attack Suite under Per-Input Query Limits

## Manuscript

`manuscript/paper.pdf` is the anonymous paper. The same folder contains its LaTeX source, `references.bib`, IEEE conference class and bibliography style, and figure files.

To compile the source, run these commands from `manuscript/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
bibtex paper
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

## Research artifact

`artifact/` contains the self-contained executable notebook, scientific Python code, frozen benchmark annotations, classifier outcome matrices, result tables, figures, and source license notices. Start with `artifact/README.md` for dependencies and reproduction instructions.

The notebook can be run independently. The script workflow uses the files together in `artifact/`.
