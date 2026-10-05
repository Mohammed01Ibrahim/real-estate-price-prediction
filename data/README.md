# Local data only

Expected file: `data/raw/pe_properties_crude.csv` (UTF-8 CSV). The private development environment used a single verified raw copy extracted without modifying either original archive. The public repository excludes it. A fresh clone requires authorized data access; no public download or redistribution license was confirmed.

You may restore the CSV from either historical ZIP with:

```powershell
python scripts/prepare_data.py --archive "C:\path\to\DS-PRO Code.zip"
# or --archive "C:\path\to\Data Science Project-Real state projects.zip"
```

This command extracts only the expected member, verifies the known SHA-256, and refuses to replace an existing different file. It never extracts arbitrary archive paths.

Required columns: `id, operation_type, currency, price_period, l1, price, surface_total, surface_covered, lat, lon, rooms, bedrooms, bathrooms, property_type, l2, l4`. Extra columns are ignored by the pipeline. `id` is checked as part of the source schema but is never a predictor. The original contains additional dates, titles, descriptions, administrative locations, and other fields. The training loader reads only this required schema; no free text, address, contact, listing duration, or end date is a feature.

106,422 original listings, all `l1=Perú`, with creation dates 2019-05-30 through 2020-06-22. Operation labels: `Venta`, `Alquiler`, `Alquiler temporal`. Currency labels include USD and PEN. Payment period is blank or `Mensual`, including monthly periods on some sale rows. Collection method, platform attribution, full schema definitions (including area units), and redistribution rights: **Not confirmed from the provided source code/data.** Area values are assumed square metres, as labeled in the historical notebook; this requires source confirmation.

Publication review found 85 rows with email-like text, 13,337 rows with phone-like text patterns (heuristic, not confirmed phone numbers), and 101,026 coordinate pairs. Names and addresses may also occur in free text; absence of these has not been established. No raw or processed records, private text, precise coordinates, or source notebook outputs are included in the public Git tree. No real sample is published; tests use generated synthetic fixtures.

`python -m real_estate.prepare_explorer` creates ignored `data/processed/explorer.json` using the same deterministic residential scope and deduplication rules as training. It keeps only validated price/area/type/counts, sanitized region/locality, and coordinates rounded to two decimals. It includes verified held-out predictions only when the source, fitted model, evaluation report, and prediction file agree. Missing coordinates remain in aggregate analytics but are omitted from map points. See [exploration architecture](../docs/exploration.md).

Code reproducibility does not grant data redistribution rights. Keep point-level exploration data local and review any future release separately.
