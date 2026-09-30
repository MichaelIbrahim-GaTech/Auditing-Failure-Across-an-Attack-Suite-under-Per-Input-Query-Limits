# Additional inference comparison and numerical certification

## Oracle calculations

The target is theta = P(S > 0). The oracle identified interval fixes the entire population count law q = H mu and optimizes theta over all latent laws compatible with q. It does not impose a parametric mixing family.

All frozen empirical frequencies are reconstructed as exact rational numbers, with their sum and round-trip error checked. Count-law equality is equivalent to agreement on all polynomial moments through degree k. For numerical support discovery, exact discrete Hahn polynomial evaluations are converted to floating point, and moment bands with radii 1e-8, 1e-10 and 1e-12 avoid numerical infeasibility at a known feasible distribution. These bands are only computational aids.

The final bounds do not rely on those bands or on LP feasibility tolerances. Every LP dual polynomial is converted to exact rational coefficients and repaired by changing its constant coefficient so that it lies below h(s), or below -h(s), at EVERY integer s from 0 through d. Its expectation under the exact rational population supplies a mathematically valid lower, or upper, endpoint bound. These checks are independent of floating-point audit-law residuals. The files retain the rational polynomial coefficients and intercepts.

When an LP support can be reconstructed as a nonnegative rational latent law matching EVERY moment through k, that law supplies an attainable endpoint witness. Its weights and support are saved. A row labeled `exact_rational_endpoint_certificates` has a total gap between primal endpoint witnesses and outer polynomial bounds below 1e-9; its decimal endpoints are therefore established to that tolerance, not asserted as symbolic exact optima. A row labeled `exact_rational_outer_bounds` has certified outer bounds but no claim that their width is sharp. The `endpoint_certificate_gap` and `width_lower` columns explicitly expose the remaining gap.

Digits at k=64 has exactly singleton identification. The observed latent support maxima are L=55 for Logistic and L=44 for Forest. Since k>L, the exact audit law assigns zero mass to C>L. Every hypothetical S>L assigns positive probability to C>L, so every compatible latent law must be supported on 0,...,L. On that support, H has full column rank: its rows C=0,...,L form an upper triangular matrix with strictly positive diagonal. The latent law and theta are consequently identified exactly. Full-suite k=d rows are also exact singletons, directly because C=S. These exact results are distinguished from merely small numerical widths.

## Linear baseline

For given d,k,m, fit coefficients g_0,...,g_k, a uniform bias bound b, and a coefficient range R by minimizing b + R/(2 sqrt(m)), subject to |sum_j H[j,s] g_j - h(s)| <= b for every s, and min g <= g_j <= max g. The fit uses neither the frozen population nor its sampled histogram. It is a bias/range regularized linear estimator, not a claim to have solved the exact minimax squared-error problem.

Write Z = m^{-1} sum_i g(C_i), and let theta_hat = clip(Z,0,1). The uniform bias is at most b. Because each g(C_i) belongs to an interval of length R, Var(Z) <= R^2/(4m). Therefore E[(Z-theta)^2] <= b^2 + R^2/(4m). Projection onto [0,1] cannot increase squared error for theta in [0,1], so the same bound holds for theta_hat. The minimized linear objective upper-bounds the square root of this MSE bound.

Hoeffding's inequality gives P(|Z-EZ| > R sqrt(log(2/delta)/(2m))) <= delta. Adding the bias bound yields the valid radius b + R sqrt(log(2/delta)/(2m)). Applying [0,1] clipping to both endpoints preserves coverage. The bias and actual coefficient range are recomputed after optimization, with a conservative floating-point allowance. The point estimate is the clipped mean of g(C), not the mean of individually clipped coefficients.

For each population, the raw estimator's mean, variance and MSE are calculated analytically from q. These are explicitly labeled `raw_*`; the raw MSE is an upper bound on the clipped estimator's MSE. Clipped bias and MSE are additionally estimated from the replicated histograms. A good point-estimation result does not establish a narrow confidence interval: the global bias/Hoeffding interval can be wider than the projection interval for these particular populations.

## Replication and uncertainty

Each setting uses 100 audit histograms. The original digits histograms and their projection intervals are reused. StrongREJECT at m=500 uses the original published seeds to reconstruct the same histogram and reuse its interval. New StrongREJECT m=100 and m=2000 histograms use fixed independent seeds and exact multinomial sampling from H mu. All means include Monte Carlo standard errors; empirical coverage includes 95% Wilson intervals. The figure uses mean width plus/minus 1.96 Monte Carlo standard errors. Oracle lines are sharp widths where certified, and explicitly marked certified upper bounds elsewhere.

The finite-sample excess over a sharp oracle width measures the combination of sampling uncertainty and conservatism of the selected interval construction. It must not be described wholly as unavoidable sampling cost or wholly as conservatism without a matching finite-sample optimality comparison.
