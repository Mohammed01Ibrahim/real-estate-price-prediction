# Methodology decisions

## Defensible target

`price` is used directly, only for `l1=Perú`, `operation_type=Venta`, `currency=USD`, blank/whitespace/missing `price_period`, and `Casa`/`Departamento`. The data does not support a verified conversion from rent to property value. Some sale records are monthly-period records; they are excluded rather than corrected by invented multipliers. A blank sale period is a provisional assumption of total advertised price. The model does not estimate rentals or realized transaction values. No historical exchange rate is assumed.

The raw dataset has a verified SHA-256 in the baseline and run metadata. The loader reads the 16 required schema fields (including two unused legacy fields); free-text/contact/date fields are not loaded. `id` is never used as a predictor. Neither source attribution nor license nor exact area definitions are confirmed. Square metres follow old notebook labels and remain an assumption.

## Deterministic cleaning and features

Sequential exclusions on the final run: 29,531 non-sale rows; 8,469 non-USD rows among remaining sales; 29,394 nonblank-period records; 16,728 unsupported property types; 10,407 missing/out-of-scope total areas; 864 exact feature/price duplicates. No additional price-invalid rows were present in the scoped data. Counts are sequential, not independent counts over the full raw CSV. 11,029 rows remain.

Required total area is bounded to 10–10,000 under a declared residential model scope. Optional bedrooms/bathrooms accept whole numbers 0–50; nonfinite or invalid values become missing during ingestion and are rejected at the public input boundary. Latitudes are bounded −19 to 1 and longitudes −82 to −68; missing/invalid/partial pairs become jointly missing during ingestion, while partial public inputs are rejected. The rectangular bounds cover the intended region but do not guarantee a point lies within Peru.

Rooms and covered area are missing in every retained observation. Ratios involving those fields were removed rather than inferred or represented as useful information. Features are eight raw predictors plus a single Haversine distance to a fixed Lima anchor. Coordinates are rounded to 0.01 degrees before geographic use. Source labels `l2` and `l4` are administrative predictors, not recreated ordinal encodings.

Optional missing numeric values use a training-fitted median with missing indicators; entirely empty fold columns retain stable shape. Optional categories use a training-fitted most-frequent value and a fitted one-hot encoder; unknown values are ignored by that encoder. There is no percentile clipping, full-data feature selection, geographic cluster fitting, end-date duration, or supervised pre-split operation. Target values remain positive finite original USD, including extreme prices.

## Split and evaluation

Proxy groups are formed without target values, using cleaned raw predictor values and rounded coordinates. Conflicting repeated source IDs assigned to distinct proxy groups fail before deduplication. Known identical-feature proxies stay together. These proxies can merge different properties and fail to link reposts whose features change; full entity resolution is not claimed. The supplied IDs are unique, so the repeated-ID guard does not alter this final experiment.

`GroupShuffleSplit` reserves 20% of groups for test; another grouped split reserves 25% of the remaining groups for validation. Actual rows are 6,625/2,209/2,195, because group sizes vary. Three-fold shuffled `GroupKFold` evaluates each entire estimator on training only. All candidates see the same folds and fixed validation partition. Median DummyRegressor, Ridge (alpha 10), HistGradientBoosting (160 iterations, 15 leaves, no internal random early-stopping split), and bounded Random Forest (100 trees, depth16, leaf4) are fixed, reproducible candidates.

Lowest validation MAE in original USD determines selection. No parameters or scope are chosen on final test results. The selected estimator is cloned and refitted on train + validation; test is used for the final result, not fitting or tuning. Nonbaseline models train in log1p space, but their estimator inverses predictions before CV and all original-dollar metrics. The baseline directly predicts the dollar median.

Validation raw-input permutation importance runs before final refitting, with three seed-controlled repeats on the training-only fitted selected estimator. Importance may shift among correlated geographic/administrative inputs and is not a causal statement. Aggregate test density/residual plots preserve the complete error range. Individual test predictions stay under ignored `artifacts/`.

## Interpretation

Random Forest's validation MAE is only 0.31% lower than boosting; the experiment does not demonstrate a statistically significant model advantage. The final test R²0.1423 and RMSE $728,516.72 expose sensitivity to extreme asking prices up to $33,218,002. The raw target is neither transaction-verified nor adjusted for current markets. High CV/validation R² does not justify treating the demo as an accurate current valuation service.

The final contract replaced an intermediate version that kept uninformative room/covered-area fields. Its group fingerprints and partition membership differ; its metrics are not comparable and are not reused. The checked-in results correspond solely to the final eight-input contract. No legacy notebook results are promoted to current evidence.

Random grouped evaluation is not a chronological or spatial extrapolation test. Future work needs confirmed schema/provenance, better entity identity, fresh licensed data, time/spatial evaluation, error investigation, and calibrated uncertainty. Code can be redistributed separately from restricted data; the existing files do not establish a dataset license.
