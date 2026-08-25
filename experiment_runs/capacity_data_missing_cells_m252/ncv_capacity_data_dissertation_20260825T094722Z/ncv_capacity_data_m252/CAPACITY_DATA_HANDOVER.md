# NCV Capacity-vs-Data Sensitivity Handover

## Experiment
- profile=dissertation
- replications=10
- monitoring_dates=252
- checkpoints=[0, 10, 25, 50, 100, 200]
- configurations=4

## Selected checkpoints by configuration
- w8_n10000: checkpoint 200
- w8_n20000: checkpoint 200
- w16_n10000: checkpoint 200
- w16_n20000: checkpoint 200

## Paired-contrast outputs
- paired contrast rows: 20
- See capacity_data_paired_contrasts.csv for means, medians, and 95% confidence intervals.

## Epoch-200 extension flag
- extension_recommended=true

## Warnings and limitations
- extension_recommended_true
- Training/validation/test/pilot are independent by construction and seeds are fixed deterministically.
- This is a sensitivity analysis rather than proof that parameter count alone caused the m=252 behaviour.
