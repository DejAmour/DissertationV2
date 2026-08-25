# NCV Capacity-vs-Data Sensitivity Handover

## Experiment
- profile=smoke
- replications=2
- monitoring_dates=252
- checkpoints=[0, 1, 2]
- configurations=5

## Selected checkpoints by configuration
- w32_n100: checkpoint 2
- w16_n100: checkpoint 2
- w8_n100: checkpoint 2
- w32_n200: checkpoint 2
- w32_n400: checkpoint 2

## Paired-contrast outputs
- paired contrast rows: 30
- See capacity_data_paired_contrasts.csv for means, medians, and 95% confidence intervals.

## Epoch-200 extension flag
- extension_recommended=false

## Warnings and limitations
- None from automated validation.
- Training/validation/test/pilot are independent by construction and seeds are fixed deterministically.
- This is a sensitivity analysis rather than proof that parameter count alone caused the m=252 behaviour.
