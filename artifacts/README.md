# Generated experiment output

`python -m real_estate.train` writes evaluation metadata, three aggregate figures, and local per-listing test predictions here. All generated outputs are ignored by Git. Reviewed aggregate figures and the verified experiment summary are copied to `docs/results/`. The private `test_predictions.csv` is never copied there. The explorer export verifies these predictions against the saved fitted pipeline before including them in the local dashboard artifact.

Every run reports filtering counts, split counts, random seed, raw CSV hash, candidate CV and validation scores, selected model, held-out results, package versions, artifact size, and validation permutation importance. Changing data or dependencies creates a new experiment; do not reuse the checked-in result claims without updating the reviewed summary.
