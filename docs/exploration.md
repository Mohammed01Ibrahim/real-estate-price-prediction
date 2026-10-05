# Exploration data, charts, and API

The estimator, map, and insights pages share a Flask application and navigation. Prediction continues through the saved fitted pipeline. Exploratory preparation calls the existing deterministic `read_raw` and `prepare_data` functions; it does not fit preprocessing, refit a model, or change evaluation partitions.

## Local preparation

After authorized data restoration and model training:

```powershell
python -m real_estate.prepare_explorer
```

Defaults resolve against the repository, including when invoked from another directory. Explicit paths are supported:

```powershell
python -m real_estate.prepare_explorer --data "C:\path\to\your.csv" --output data/processed/explorer.json --model models/pipeline.joblib --evaluation artifacts/evaluation.json --predictions artifacts/test_predictions.csv
```

The ignored JSON contains a schema version, source SHA-256, scope, generation date, retained-row count, coordinate precision, allowlisted listings, and an evaluation block. An atomic replacement prevents half-written artifacts. The exporter validates the complete artifact, including nested evaluation fields, before saving it. It does not publish data. A fresh public clone contains the code and instructions, rather than data or a fitted model.

Missing evaluation inputs allow exploration to work with model diagnostics explicitly unavailable. When supplied, saved evaluation data must agree with the source hash, fitted model metadata, deterministic split, and predictions recomputed with that exact pipeline. Stored test metrics are checked against the verified test arrays. A mismatch produces the same safe unavailable state, without falling back to historical scores. Regenerate the explorer after a new training run.

## Privacy and scope

Each record contains only `price_usd`, `surface_total`, `bedrooms`, `bathrooms`, `property_type`, `l2`, `l4`, `lat`, and `lon`. There are no source IDs, descriptions, titles, dates, street addresses, sellers, accounts, names, emails, or phone fields. Administrative labels containing control characters, email-like content, long phone-like digit sequences, or excessive length become unknown. Coordinates are rounded to 0.01 degrees and explicitly displayed as approximate.

The original residential scope remains authoritative: Peru, Venta, USD, blank payment period, Casa/Departamento, positive finite price, and total area between 10 and 10,000 m². Invalid optional values become missing through existing preparation rules. Listings without valid paired coordinates remain in analytics and are counted separately; they never become fabricated map points. Broad coordinate bounds cannot prove that a point lies within Peru.

Point-level data and test arrays remain local and ignored by Git. Coordinate rounding is not anonymization or permission to redistribute records. Bind the app to localhost for the documented workflow. Any deployment exposing these endpoints needs a separate rights/privacy decision.

Leaflet scripts, styles, MarkerCluster, and Plotly are served from pinned local assets. The basemap requests tiles from OpenStreetMap, which receives the browser's IP address and requested map tiles. It does not receive the listing JSON from this application. Tile loading requires internet; the app's filters, charts, data requests, and markers use the local Flask server. Preserve visible attribution and follow the [OpenStreetMap tile policy](https://operations.osmfoundation.org/policies/tiles/); no offline tile downloading is implemented.

## Routes and filters

| Route | Behavior |
|---|---|
| `/` | Estimate form using the complete saved pipeline |
| `/explore` | Clustered listing map and property details |
| `/insights` | Interactive analytics and fixed model diagnostics |
| `POST /api/predict` | Existing strict prediction contract |
| `GET /api/listings` | Bounded page of mapped records and completeness counts |
| `GET /api/insights` | Filtered aggregates, bounded scatter, fixed evaluation |
| `GET /api/options?l2=Lima` | Region and region-scoped locality options |
| `GET /health` | Fitted model readiness; independent of exploration readiness |

Both exploratory APIs accept `property_type`, `l2`, `l4`, `q`, `min_price`, `max_price`, `min_area`, `max_area`, `bedrooms`, and `bathrooms`. Exact category/count filters and inclusive numeric ranges are applied by Python. `q` searches administrative labels literally, case-insensitively. Text is bounded to 100 characters, counts must be integers from 0 to 50, prices nonnegative finite numbers, areas within the project scope, and minimums cannot exceed maximums. Unknown/repeated parameters and invalid values return safe HTTP 400 errors. Missing/corrupt explorer artifacts return safe 503 errors referring to README setup; the map also displays the preparation command. Unexpected errors are logged server-side and return a generic 500.

`/api/listings` additionally uses `limit` (1–2,000; default 2,000) and `offset` (0–250,000), returning `next_offset`, total matching count, mappable count, missing-coordinate count, and returned count. The map retrieves all pages, verifies completeness, and adds markers in batches between animation frames. It aborts stale requests and discards incomplete failed results. Numbered clusters zoom/spiderfy into individual points. Hover previews show asking price, type, total area, counts, and region/locality. Click details add the rounded coordinate pair; missing fields display Unknown.

The shared parser also accepts pagination parameters for insights, but they do not reduce aggregate populations. Dashboard UI does not send them. Options accept only optional `l2`.

## Analytical story

All summary cards and exploratory charts reflect the selected records. Price and area histograms use their full observed ranges without tail clipping. The optional log price view uses geometric bin edges in actual USD. Type counts use only Casa/Departamento. Regional comparisons rank listing counts and show median asking prices, with population counts available on hover. Missing region labels are displayed as Unknown.

The price–area scatter uses at most 2,000 deterministically sampled records, explicitly labels sampling, and preserves minimum/maximum price and area observations. It is a display sample; summaries and histograms use every selected listing. Scatter hover contains only allowlisted property/context fields. Pearson correlations use valid area, bedroom/bathroom counts, rounded latitude/longitude, and raw USD price, with pairwise available observations. Hover reports paired counts. Undefined correlations remain blank. No categorical numeric codes or removed historical variables are included. Correlation does not imply causation.

Final held-out predicted-vs-actual, residual scatter, and residual histogram use every verified test prediction. Residual means actual minus predicted USD, so positive values indicate underestimation. Linear/log prediction axes are optional. Evaluation is independent of exploration filters: filtering listings must not change a reported test population or score.

Permutation importance displays the saved validation-stage scores for the selected pipeline, measured before its development refit, with standard-deviation error bars. It is not new importance measured on the final test set. Shuffling correlated features can share importance or create implausible combinations; the chart describes predictive reliance, not causal effects. Old notebook importance images and obsolete scores are excluded.

## Performance and dependencies

The raw CSV is processed once during preparation. HTTP requests read a schema-validated derived artifact, cached by modification time and size, and maintain a bounded 32-response cache that resets on artifact replacement. Artifact size is limited to 100 MB and 250,000 listings; each map response contains at most 2,000 points. The map uses [Leaflet 1.9.4](https://leafletjs.com/reference.html) and [MarkerCluster 1.5.3](https://github.com/Leaflet/Leaflet.markercluster). Charts use Plotly.js 3.1.0. No additional Python runtime dependency or paid API is required.

Vendor files, licenses, URLs, sizes, and SHA-256 hashes are recorded in `static/vendor/manifest.json`. Verify them offline with:

```powershell
python scripts/vendor_frontend.py --verify
```

Maintainers can deliberately redownload the pinned assets with `python scripts/vendor_frontend.py`; normal installation and application startup do not contact JavaScript CDNs.
