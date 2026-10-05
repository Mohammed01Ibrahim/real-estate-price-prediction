# Verification and publication review

Performed locally on Windows, Python 3.14.7, 2026-10-05. Nothing was pushed or published.

| Check | Evidence / outcome |
|---|---|
| Environment | Created a new empty `.venv`; installed the declared requirements; recorded exact direct/transitive runtime/test pins; reinstalled to validate direct pins. `pip check` passed. |
| Core checks | `python -m compileall -q src scripts` and `git diff --check` passed. |
| Automated behavior tests | `python -m pytest`: **112 passed**, final recorded run 2.84 seconds. Synthetic fixtures need neither raw dataset nor pretrained model. |
| Full real-data training | All four candidates completed three training-only CV folds and validation scoring on the final supported scope. Selected Random Forest was refitted on development data, serialized, reloaded, and evaluated on 2,195 test rows. |
| Artifact | Compressed complete pipeline: 4,887,387 bytes. Serialization/inference equality tested. Missing, corrupt, unfitted, and schema/library-incompatible artifacts fail safely. |
| Final consistency | After the repeated-ID guard was added, final partitions and stored model metrics were recomputed and matched the recorded results without refitting. Supplied source IDs are unique. |
| Flask startup | `python -m real_estate.web` started on 127.0.0.1:5000 with debug/reloader disabled. HTML, CSS, and health returned HTTP 200. |
| Live API | Three invented properties produced finite positive estimates ($165,546.51, $227,259.16, $105,367.76). Five invalid requests returned 400: negative area, unsupported type, incomplete coordinate pair, boolean count, obsolete room field. |
| Additional app behavior | Tests exercise forms, malformed/non-object JSON, body size limits, input escaping, missing model 503, and safe unexpected-error responses. |
| Browser | Actual form submitted with invented details and omitted coordinates, producing $165,547 when rounded for display. Saved screenshot is public and contains no real listing. |
| Data preparation | Script extracted and SHA-256-verified the CSV from the original ZIP to a disposable temporary directory. Existing correct file is left alone; different existing content is refused. |
| Portability | Default model prediction and training CLI import/help succeeded from the parent working directory. Tests also change working directory. No original notebook/artifact/dataset-driven app dropdown is required. |
| Notebook | Generated clean optional EDA notebook; JSON/schema inspected, outputs cleared. Notebook was not executed interactively. |
| CI | Configured Windows/Linux, Python 3.12/3.14, dependency installation, package consistency, compilation, and synthetic tests. Remote matrix not executed because nothing was pushed. |

Windows virtual-environment creation, dependency installation, training, tests, and app startup were executed. Activation is optional; all verification used the explicit `.venv\Scripts\python.exe` executable, so it does not depend on the user's PowerShell execution policy. Other Python/OS combinations are CI targets, not locally verified claims.

The public candidate tree was scanned for common credential/token/private-key patterns, user-specific absolute paths, debug enabled, binary datasets/models, large files, and notebook outputs. No actual credential, personal path, enabled debugger, model binary, private dataset, or output-bearing notebook was found. This is a scoped publication review, not a proof that every conceivable secret pattern is absent.

`git check-ignore` confirmed raw CSV, final fitted model, individual test predictions, and environment metadata are excluded. Raw free text has email-like and phone-like content and source coordinates; all source/processed records and old notebook outputs remain private. Only aggregate experiment figures/summary and an invented-property screenshot are tracked. The raw collection license and source attribution remain unconfirmed; no data or model redistribution rights are asserted.

The original archives remain outside the newly initialized repository. No unrelated repository was modified. Git history records Phase 2 work as it happened; no remote was configured. No LICENSE was fabricated. The owner may choose a code license after checking original ownership/attribution; this would not license the dataset.

The repository is ready for **code-only portfolio publication**, with the README assumptions and performance limitations retained. There are no remaining implementation blockers for the documented local workflow. Data/model publication requires a separate rights/privacy review. Current-market valuation, source-schema confirmation, future-time/spatial performance, and calibrated uncertainty remain outside the verified scope.
