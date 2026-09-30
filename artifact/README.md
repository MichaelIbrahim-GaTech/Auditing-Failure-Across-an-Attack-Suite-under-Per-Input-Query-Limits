# Auditing Failure Across an Attack Suite under Per-Input Query Limits

This artifact contains the code, data, and results accompanying the paper.

## Run the analysis

The self-contained `SaTML_Attack_Suite_Audit.ipynb` includes the analysis code and compact frozen benchmark inputs. Open it in Jupyter and run the cells in order. It writes its results to `satml_audit_outputs/`. The supplied notebook includes executed outputs.

Use Python 3.12 with the scientific packages listed in `requirements.txt`. Install them in a virtual environment before running the notebook:

```bash
python -m pip install -r requirements.txt
```

Jupyter is needed only to use the notebook interface. The analysis runs on a CPU, makes no live model calls, and requires no network access once its dependencies are installed. Runtime depends on the machine.

The same analysis can be run from this directory using the supplied scripts:

```bash
python experiments/numerical.py
python empirical.py
python run_confidence.py
python jbb.py
python jbb_intervals.py
python strongreject.py
python run_strongreject_audit.py
python zero_detection.py
python decision_example.py
python decision_consequences.py
python run_heldout_ordering.py
python run_oracle.py
python run_completion.py --repetitions 300
python validate_completion_theory.py
python run_interpretation.py
python render_figures.py
```

`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` can reduce threading overhead. Scripts write their outputs to the corresponding result folders. Numerical differences can arise with different scientific-library versions.

## Files

| Files or folder | Contents |
|---|---|
| `audit_data.py` | Loads the six frozen outcome populations. |
| `audit_inference.py`, `confidence.py`, `completion_audit.py` | Inference and audit estimators. |
| `experiments/` | Finite-suite identification calculations and rational witnesses. |
| `empirical_results/` | Digits data splits, classifier outcomes, and audit trials. |
| `jbb_results/`, `strongreject_results/` | Published benchmark annotations, source information, and upstream licenses. |
| `confidence_results/`, `strongreject_audit_results/`, `oracle_results/` | Confidence intervals, population identification bounds, and comparisons. |
| `completion_results/` | Measured and controlled experiments, trial tables, study design, and figures. |
| `zero_detection_results/`, `decision_results/`, `consequence_results/` | Exact calculations for zero detections and constructed certification examples. |
| `ordering_results/` | Pilot splits, learned orders, held-out results, and study protocol. |
| `interpretation_results/` | Attack overlap, cost, variance, and threshold-sensitivity summaries. |
| `figures/` | Additional generated figures. |

The rational certificates and result-level numerical checks document the calculations used in the paper. They are distinct from the reported empirical measurements.

## Data and interpretation

StrongREJECT and JailbreakBench results use archived public annotations. Source URLs, release identifiers, citations, and license notices are included in their respective folders. The artifact contains scalar annotations and record identifiers, not attack prompts or generated responses. The digits experiment uses the public dataset bundled with scikit-learn.

Constructed low-prevalence populations are synthetic. The digit perturbations are wrong-label stress tests and can change image semantics. Repeated audit trials measure sampling variation over the fixed populations. The four security matrices share contexts, and the two classifier matrices share the digits dataset.

The confidence comparisons evaluate the implemented methods. They do not establish that one allocation is universally best under every valid confidence procedure. The learned-order experiment conditions on one pilot split; its measured savings do not establish deployment transfer.
