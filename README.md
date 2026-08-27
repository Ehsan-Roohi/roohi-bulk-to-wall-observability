# Quantity-Dependent Bulk-to-Wall Observability

Research code and processed reproducibility artifacts for studying quantity-dependent bulk-to-wall observability in rarefied hypersonic flow over a surface protrusion.

The workflow combines DSMC-derived flow fields, coordinate-conditioned neural surrogates, nonlocal bulk-to-wall diagnostics, closed-loop checks, and tree-bootstrap/alternative-regressor robustness controls.

## Authors and contributions

- **Ehsan Roohi** — conceptualization, machine-learning methodology, software, analysis, visualization, writing, and supervision.
- **Elyas Lekzian** — conceptualization, DSMC data curation, geometry verification, rarefied-flow methodology and interpretation, validation, and review.

Please use [`CITATION.cff`](CITATION.cff) when citing this software release.

## Repository contents

- `src/`: 12 research scripts for surrogate training, physical post-processing, observability analysis, closed-loop assessment, uncertainty/robustness controls, and figure generation.
- `data/processed/`: processed tables, manifests, run configurations, and summary outputs used in the analysis.
- `figures/`: selected reference figures for verification.
- `CHECKSUMS.sha256`: integrity hashes for the public release files.

Raw DSMC field archives, model checkpoints, private correspondence, review files, and manuscript working files are intentionally not included.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation can be adjusted for the local CPU/GPU platform by following the official PyTorch installation selector.

## Reproduce figures from the included processed data

Generate the surface-profile figure:

```bash
python src/21_make_fig09_surface_profiles_publication.py \
  --in data/processed \
  --out reproduced_figures/surface_profiles
```

Generate the conceptual framework figure:

```bash
python src/20_make_framework_figure_clean_v2.py \
  --out reproduced_figures/framework
```

These two commands operate without the raw DSMC archives. The training and raw-field analysis scripts require separately supplied DSMC archives and, for some legacy stages, historical base modules/checkpoints that are not part of this public release. The public manifests use portable placeholder paths under `raw_data/`; users with authorized raw data should update those paths locally.

## Scientific scope

The primary diagnostic uses a 27-case factorial block spanning Mach number, Knudsen number, and protrusion orientation. Additional height and wall-temperature cases support generalization studies. The reported radius is a model-conditional predictive-support length, not a causal or constitutive material scale.

The processed package records 69 raw-file entries and 57 unique physical conditions after duplicate-key auditing. Raw DSMC fields remain available from the corresponding author subject to the associated study's data-access terms.

## Licensing

- Source code: [MIT License](LICENSE)
- Processed data and reference figures: [CC BY 4.0](DATA_LICENSE.md)

Third-party dependencies retain their own licenses.

## Contact

For scientific questions or access to the underlying raw DSMC data, open a GitHub issue or contact the corresponding author through their institutional profile.
