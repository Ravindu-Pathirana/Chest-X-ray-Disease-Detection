# T45 presentation

`T45_research_progress.pptx` is a 10-slide progress deck generated from the
committed CNN closeout, paired EIL and preliminary T35 architecture-only CPU
artifacts. It intentionally labels perturbation, RSNA and final GPU efficiency
results as unmeasured.

Regenerate from the repository root after installing
`requirements-presentation.txt`:

```powershell
py -3.13 scripts/build_t45_slides.py
```

The source script reads `artifacts/cnn_closeout/cnn_multiseed_summary.csv`,
`artifacts/explainable_ai/paired_eil_bootstrap.csv`, and
`artifacts/T35_efficiency/architecture_only_efficiency.csv`. Refresh the deck
only after those sources are verified. Do not replace pending results with
illustrative values.
