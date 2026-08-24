# Stage 8 Handover

- profile: m252_n20000
- base seed: 42
- replications: 5
- monitoring dates: 252
- hidden width: 32
- trainable parameters: 8129
- NCV training paths: 20000
- NCV checkpoint: 200
- experiment role: expanded_training_followup_stage8_m252_n20000_r5
- sensitivity interpretation: primary Stage 8 default configuration
- fixed NCV epoch: 200 (capacity_and_data_sensitivity_best_tested_configuration)
- neural_cv.train_network default epoch count: 200 (generic function default; Stage 8 overrides to fixed 200)
- Stage 8 uses fixed checkpoint from validation-based training-curve study; final test/pricing does not select epoch online
- Torch: available (2.13.0+cu130)
- failed-row count: 0
- successful-row count: 200
- empirical equal-budget mode run: False
- reproducibility tested through second identical run: False

## output files
- aggregate_results.csv
- aggregate_statistical_results.csv
- break_even_equal_observations.csv
- break_even_fixed_accuracy.csv
- break_even_matched_accuracy.csv
- config_snapshot.json
- environment.json
- equal_budget_empirical_results.csv
- equal_budget_projected_results.csv
- equal_pricing_observations_summary.csv
- high_precision_references.csv
- matched_accuracy_results.csv
- per_replication_results.csv
- per_replication_variance_ratios.csv
- portfolio_break_even.csv
- reference_precision_diagnostics.csv
- reproducibility_report.json
- runtime_raw.csv
- runtime_summary.csv
- seed_manifest.csv
- shared_reference_training.csv
- summary_stable.csv
- transfer_diagnostics.csv
- transfer_diagnostics_summary.csv
- validation_report.json
- variance_ratio_summary.csv

## reproducibility note
- stable summary hash is available for run-to-run comparison; reproducibility is only confirmed after a second identical run matches.
