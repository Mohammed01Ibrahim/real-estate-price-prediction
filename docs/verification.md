# Verification and publication review

Verified locally on Windows with Python 3.14.7, 2026-10-05. The application retains the existing fitted pipeline and verified experiment; exploration work did not retrain or change it. Nothing was pushed or published.

## Preserved model evidence

The recorded training run completed four fixed candidates, three training-only group CV folds, validation selection, development refit, serialization, reload, and evaluation on 2,195 held-out rows. Training and validation contain 6,625 and 2,209 rows. The selected Random Forest pipeline remains 4,887,387 bytes. Held-out MAE is $90,684.97, RMSE $728,516.72, and R² 0.1423. These limitations remain visible in the README and dashboard.

Explorer preparation independently verified the source hash, saved model/run metadata, every split's row/group counts, final predictions from the loaded pipeline, and metrics recalculated from those arrays. Saved validation-stage permutation scores are distinguished from final held-out results. No legacy notebook score, residual plot, or impurity-importance image was reused.

## Application and data checks

| Check | Evidence / outcome |
|---|---|
| Preserved tests | All 112 earlier tests remain; no pipeline/inference tests were removed. |
| New tests | 93 exploration/analytics regressions cover filters, privacy allowlists, sanitized labels, paired rounded coordinates, pagination, artifact caching/reload, missing/corrupt files, full-tail histograms, medians/counts, pairwise correlations, sample bounds/extrema, and mismatched evaluation metadata/predictions. |
| Complete suite | **205 tests passed** in the working repository (7.31 seconds). Fixtures require neither real data nor a pretrained model. |
| Clean public export | Exported committed application/test code (`000488b`) into a disposable directory, created an independent empty Python 3.14.7 environment, and installed exact requirements. Its own source imported correctly; `pip check`, compilation, 14 vendor hashes, and **205 tests passed** (7.67 seconds). Private CSV, model, explorer JSON, test predictions, and original archives were absent. Three pages and all 20 vendor/app assets returned 200; missing-data/model APIs returned safe 503 and debug was disabled. Subsequent edits concern documentation/screenshots only. |
| Prepared scope | 11,029 retained listings, 23 reported regions, median price $198,000, median area 140 m², 3,777 Casa and 7,252 Departamento. All six values independently matched existing `prepare_data` output. |
| Map completeness | 10,707 mappable coordinate pairs and 322 missing-coordinate listings. Browser loaded all six API pages and reported the complete map population. Missing coordinates never become fabricated locations. |
| Model consistency | Explorer export rechecked predictions from the exact saved pipeline and included verified evaluation. Model size and held-out metrics stayed unchanged. |
| Missing-data browser | Both exploration pages displayed safe setup guidance with missing private artifacts; chart/summary values cleared instead of displaying stale results. The map showed the preparation command. |
| Flask startup | `python -m real_estate.web` runs on 127.0.0.1:5000 with debugger/reloader disabled. All three pages and local CSS/JavaScript assets load. |
| Estimator browser | Invented Departamento, 120 m², 3 bedrooms, 2 bathrooms, Lima, unseen demo locality, no coordinates: form displayed USD 165,547, matching the preserved pipeline's earlier result. |
| Map browser | Hover showed price/type/area/counts/location; click populated richer details with approximate coordinate labels and Unknown missing values. Casa + Lima matched 2,447 listings / 2,367 mapped points. Empty search cleared points; reset restored all results. Cluster clicks zoomed into points. No JavaScript warnings/errors observed. |
| Insights browser | Eleven interactive plots rendered. Hover labels, log views, Casa filtering, empty/reset recovery, invalid filter states, and fixed global test diagnostics checked. Casa filtering produced 3,777 listings, median $287,000 and 240 m²; model scores stayed fixed. Undefined correlations remain blank with paired-count context. |
| Responsive layout | Insights checked at 390 × 844: no horizontal page overflow, charts stayed within cards, summaries used two columns and evaluation cards one column. Map interaction verified at ordinary desktop width. |
| Core/package checks | `pip check`, `compileall`, and `git diff --check` passed. Fourteen pinned vendor downloads passed offline SHA-256/size verification. Git attributes preserve exact vendor bytes across Windows checkouts. |
| Publication scan | Scanned the candidate Git tree for credential/token/private-key patterns, user-specific Windows paths, debug enabled, private data/model/archive/environment files, oversized artifacts, and notebook outputs. No actual secret, private listing text, personal path, or output-bearing notebook found. The largest file is licensed Plotly.js, 4,763,993 bytes. Dataset contact fields are excluded; vendor author/license contacts are legitimate third-party notices. |
| Notebook | Optional notebook remains output-free; professional text and cell IDs replaced internal workflow terms. It was not executed interactively. |
| CI | Windows/Linux × Python 3.12/3.14 matrix includes dependency installation, package consistency, compilation, offline vendor verification, and synthetic tests. Remote CI has not run because nothing was pushed. |

The earlier verified archive extractor remains unchanged: it extracts only the expected CSV, checks SHA-256, preserves existing matching content, and refuses different content. Default paths remain portable. Activation is optional; verification uses the explicit virtual-environment executable without requiring an execution-policy change.

## Public contents

Raw CSVs, processed point-level JSON, fitted models, individual test predictions, original ZIPs, environments, caches, and temporary verification/demo files remain ignored. Public map screenshots use generated fictional properties with demo locality labels. The dashboard screenshot contains aggregate summaries/distributions; the estimator screenshot uses invented input. Reviewed aggregate evaluation figures and summary remain under `docs/results/`.

Source schema details, attribution, and redistribution rights remain **Not confirmed from the provided source code/data.** Coordinate rounding reduces precision but does not anonymize listings or grant publication rights. The documented app binds to localhost; publishing its code does not authorize deployment of private data endpoints.

Current files use professional project language. Earlier Git commits were preserved with their original messages; history was not rewritten. No collaborators, experiments, or historical contributions were invented. Third-party libraries retain their licenses. A project code license requires the owner's attribution/ownership decision and would not license the dataset.

The code is intended for portfolio publication with its stated data-access requirements and performance limitations. Current-market valuation, confirmed target/unit semantics, temporal/spatial generalization, and calibrated uncertainty remain outside the verified scope.
