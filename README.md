# Worse Than the Local Mean: transferring satellite maize yield models across African countries

Code, results and manuscript source for the ICICT 2027 paper on whether a field level maize
yield model trained in some Sub-Saharan African countries can be deployed in another, how many
local labels repair it, and when a transferred model should not be deployed.

## What the pipeline does

1. **Labels.** Maize yield records from the GROW-Africa database with GPS coordinates
   (One Acre Fund surveys, the Niche on farm trials, LSMS household surveys) for Kenya, Malawi,
   Nigeria, Rwanda and Tanzania, harvest years 2017 to 2021. Up to 1,600 fields per country
   are drawn with an equal quota per harvest year and national yield quintile, shortfalls filled
   at random (seed 42).
2. **Imagery and rainfall.** For each field, a 64 x 64 pixel (640 m) Sentinel-2 Level-1C median
   composite over its growing season, masked with Cloud Score+ (`cs_cdf` at least 0.60), and
   CHIRPS daily rainfall over the same window. The window runs from the recorded planting date to
   the recorded harvest date when both exist and lie 60 to 270 days apart, otherwise 150 days before harvest, otherwise the
   national crop calendar.
3. **Features.** Ten band medians, NDVI, EVI, LSWI and NDWI, and three rainfall statistics
   (17 features), plus frozen CLS embeddings from Prithvi-EO-1.0-100M and ViT-Base/16.
4. **Experiments** on two analysis sets: `precise` (fields whose GROW-Africa spatial precision is
   30 m; Kenya, Malawi, Rwanda, Tanzania) and `all` (every field, five countries). Every split is
   grouped by location because 3,416 of the 7,379 fields share coordinates with another field
   (`results/label_audit_locations.csv`).

| Script | Experiment | Main outputs |
|---|---|---|
| `05_transfer.py` | E1 leave one country out, E2 within country reference, E4 error decomposition | `results/transfer_scores.csv` |
| `06_label_audit.py` | E3 splitting design and label geolocation precision | `results/label_audit_*.csv` |
| `07_few_shot.py` | E5 calibration with k local labels | `results/few_shot_summary.csv` |
| `08_shift_diagnostic.py` | E6 label free shift score across deployment scenarios | `results/shift_*.csv` |
| `09_decision_table.py` | E7 deployment decision table | `results/decision_table.csv` |
| `10_figures.py`, `11_tables.py` | figures and LaTeX tables for the paper | `figures/`, `results/tables/` |

In the paper: Fig. 1 is `figures/fig1_label_audit`, Fig. 2 is `figures/fig2_few_shot`, and
Tables 1 to 3 are `results/tables/data.tex`, `transfer.tex` and `decision.tex`. Two outputs
are supplementary and not printed in the paper: `figures/fig3_shift` (shift score against the
transfer to within country RMSE ratio, both analysis sets) and `results/tables/features.tex`
(transfer and within country R2 per feature set).

The decision rule in `09_decision_table.py` covers transferred models (as is, or after offset,
linear or pooled calibration). A ridge model fitted to the local fields alone is reported next to
it (`r2_k200_local_ridge`, `r2p05_k200_local_ridge`) but is not part of the rule.

## Reproducing the results

Python 3.14 on CPU. Install the pinned packages:

```
pip install -r requirements.txt
```

Data access (not redistributed here):

* GROW-Africa: download `GROW-Africa-Database.zip` from the data paper's repository into `data/raw/`.
* Google Earth Engine: an account and a cloud project (`earthengine authenticate`), used for
  `COPERNICUS/S2_HARMONIZED`, `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` and `UCSB-CHG/CHIRPS/DAILY`.
* Prithvi-EO-1.0-100M weights: `Prithvi_EO_V1_100M.pt` from Hugging Face into `models/`.
  ViT-Base/16 (`google/vit-base-patch16-224`) is fetched by `transformers`.

Then run, in order:

```
python scripts/01_extract_labels.py
python scripts/02_extract_earth_engine.py      # about 25 min with 16 threads; resumable
python scripts/03_build_features.py
python scripts/04_extract_embeddings.py --model prithvi
python scripts/04_extract_embeddings.py --model vit
python scripts/05_transfer.py
python scripts/06_label_audit.py
python scripts/07_few_shot.py --workers 4
python scripts/08_shift_diagnostic.py
python scripts/09_decision_table.py
python scripts/10_figures.py
python scripts/11_tables.py
python -m pytest tests
```

Every random draw is seeded, so a rerun on the same data reproduces the tables. Earth Engine
can return slightly different composites if a collection is reprocessed upstream.

Every long step saves its progress and resumes when rerun after an interruption: Earth Engine
extraction and embeddings per field, transfer predictions per configuration
(`logs/transfer_cache`), few shot draws per configuration and target (`logs/few_shot_cache`)
and shift scenarios per target (`logs/shift_cache`). Delete a cache folder to recompute it.

## Repository layout

```
src/cropshift/   earth_engine, features, embeddings, datasets, models, evaluation,
                 calibration, shift, plotting, prithvi_mae (Apache 2.0, from IBM and NASA)
scripts/         numbered pipeline steps
results/         every number reported in the paper (CSV) and generated LaTeX tables
figures/         figures as vector PDF and 300 dpi PNG
paper/           LaTeX source (Springer svproc class)
tests/           unit tests (pytest)
```


## Use of AI tools

Analysis code in this repository was developed with the assistance of an AI coding assistant
(Claude, Anthropic). The authors reviewed the code and results and take full responsibility
for them.

## Licence

Code: MIT (see `LICENSE`). Third party components and data: see `NOTICE`.
