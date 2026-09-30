# Total-budget zero-detection result

Run from the package root:

```bash
python zero_detection.py
```

The module uses only the Python standard library, needs no data files or network,
and finishes the predetermined checks in about one second on the development
runtime. `--output-dir PATH` changes the destination. Importing the module does
not write files.

`allocation_checks.csv` compares the closed expression against two independent
finite formulations in 160 cases: every integer depth allocation using at most
the budget, and a Bellman recursion that can start, return to, or stop querying
inputs along the all-zero transcript. All arithmetic is rational. Cases cover
`d=1,...,4`, `B=0,...,7`, and prevalence `0, 1/5, 1/2, 4/5, 1`. The enumeration
checks 1,550 allocation products; the Bellman calculations evaluate 1,449 states.
Input permutations and coordinate-label copies are omitted because they have
identical probabilities under the symmetric singleton population.

`validation.json` records the result and the module's SHA256. Additional checks
directly enumerate uniform query subsets to verify attainment over every
positive success-count vertex at six small suite sizes, verify inverse-bound
boundary cases, and check exact root brackets and outward conversion to floats.
These finite checks support the general written proof; they do not enumerate
all possible infinite policy classes.

`bound_examples.csv` records nine predetermined boundary cases and budgets
already used in the manuscript. The root is bracketed with exact fractions to
width at most `2**(-80)`; the displayed upper bound rounds outward. At `d=64`,
`B=2000`, and error probability `.05`, the complete-then-partial allocation gives
a zero-success upper bound of about `0.09143657`. Using only 31 complete labels
gives `0.09211406`; uniform depth-four screening of 500 inputs gives `0.09557682`.

The optimal design determines complete labels on `floor(B/d)` independent
inputs, then uses any remainder on a uniform subset of another input. Early
stopping at success preserves its zero-success event and hard budget ceiling.
The corresponding common nonrandom endpoint can be reported after no successes,
with `[0,1]` otherwise, to obtain coverage of at least `1-delta`.

The endpoint must not be attached automatically to arbitrary audits sharing the
same budget ceiling. The theorem optimizes worst-case no-success probability
at fixed prevalence, and the common deterministic endpoint on that event. It
does not optimize a complete confidence procedure, mean squared error, or
expected expenditure. No literature-novelty claim is made for this extension.

The manuscript gives the theorem, proof, inversion argument, and boundary cases.
