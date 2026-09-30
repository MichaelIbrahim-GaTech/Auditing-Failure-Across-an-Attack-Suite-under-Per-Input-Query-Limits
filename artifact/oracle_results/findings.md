# Population identification and finite-sample results

The observed confidence-projection widths substantially exceed the population identification widths in several settings. This is especially pronounced at k=16. It would be incorrect to describe the full observed width as an irreducible information floor.

| Population | k | Sharp oracle width or certified upper bound | Projection mean width, m=100 | m=500 | m=2000 |
|---|---:|---:|---:|---:|---:|
| GPT-4o-mini | 4 | 0.096565, sharp | 0.46110 | 0.34033 | 0.27785 |
| GPT-4o-mini | 16 | <=0.000204514 | 0.17119 | 0.09995 | 0.07194 |
| Forest | 4 | 0.456595, sharp | 0.86791 | 0.77197 | 0.71060 |
| Forest | 16 | <=0.0215862 | 0.63811 | 0.53533 | 0.47221 |

At StrongREJECT k=4, the sharp widths are 0.00956303 (Dolphin), 0.0228223 (GPT-3.5), 0.0965647 (GPT-4o-mini), and 0.0743301 (Llama-3.1). At k=16, the certified upper widths are 0.00000195614, 0.000135361, 0.000204514, and 0.000207148 respectively. Only the Dolphin k=16 bound is accompanied by endpoint witnesses proving sharpness to the stated numerical tolerance. The other three k=16 widths are upper bounds.

Digits at k=64 is exactly identified for both models, using the finite-support argument in inference_notes.md. Logistic at k=16 has certified oracle width at most 0.0333622; Forest has width at most 0.0215862. At k=1, their sharp widths are 0.913045 and 0.952895, respectively. Thus the broad low-budget intervals do include substantial nonidentification, whereas the much broader finite intervals at k=16 or k=64 cannot be explained by that alone.

The optimized bias/range linear estimator is a credible, population-independent partial-audit baseline, but its global bias plus Hoeffding intervals are not uniformly competitive here. They are narrower than the original projection in only 2 of 66 settings, and every one of the 66 empirical coverage proportions is 1.00 in 100 repeats. This high empirical coverage is compatible with substantial conservatism; it does not establish exact coverage of 1 or optimality of either procedure. At 100/100 covered repetitions, the Wilson 95% lower confidence limit is approximately 0.963.

The point estimator's analytic raw MSE and empirical clipped MSE are stored separately. Its valid interval and point-estimation performance should not be conflated. The linear objective minimizes an upper bound on root mean squared error, rather than expected confidence-interval width, so finite-sample interval widths need not decrease monotonically when its fitted coefficients change with m.

All 66 settings use 100 independent audit replications. Mean-width standard errors and Wilson intervals for coverage appear in finite_summary.csv. The four-panel figure compares the prespecified GPT-4o-mini and Forest populations at k=4 and k=16, with m=100,500,2000. Error bars are 1.96 Monte Carlo standard errors. The difference above a sharp oracle width reflects sampling uncertainty plus conservatism of the chosen confidence construction; no finite-sample minimax decomposition is claimed.
