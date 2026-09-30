# Frozen-label JailbreakBench case study

Source: [JailbreakBench artifacts](https://github.com/JailbreakBench/artifacts), pinned commit `909e68c01d94222b8ad2e397a017e2e12e2adb73`. The source manifest records each artifact URL and SHA256. The upstream MIT notice is included. Cite Chao et al., *JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models*, NeurIPS Datasets and Benchmarks 2024, [arXiv:2404.01318](https://arxiv.org/abs/2404.01318).

The processed data retain the published Boolean `jailbroken` field unchanged. The separate historical `jailbroken_llama_guard1` field is retained for traceability and is never substituted for it. No annotation was corrected. All ten Privacy-category behaviors are excluded, leaving 90 aligned behaviors per artifact. No prompts or responses are included.

The primary suite contains the four methods present for every target model: GCG, JBC, PAIR, and `prompt_with_random_search`. An additional analysis includes DSN only where its artifact exists. Every matrix is restricted to the intersection of behavior IDs observed for its methods. No absent label is imputed. Blank method columns mean the method is outside that suite. `S` is the number of positive labels in that suite.

This is a retrospective analysis of fixed, historical annotations, not a new evaluation of these models or of current deployed systems. A queried matrix entry represents a complete published strategy outcome. It is not one LLM API call or an independent Bernoulli retry. Upstream strategies used different access assumptions and optimization budgets, recorded in the compact metadata and source manifest. The analysis does not compare the efficiency or present safety of different models.

Run `python jbb.py` from the parent directory to regenerate all summaries from the shipped metadata. Add `--download` to fetch and hash-check the pinned upstream files again; only filtered metadata are written. `summary.json` contains exact success fractions, histograms, and exact hypergeometric count distributions derived from the full matrices. Those distributions are oracle references, not estimates from a limited audit.
