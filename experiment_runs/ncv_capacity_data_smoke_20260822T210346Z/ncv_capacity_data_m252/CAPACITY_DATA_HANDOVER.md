# NCV Capacity-vs-Data Sensitivity Handover

## Exact experiment completed
Additive sensitivity run for m=252 on the reference arithmetic Asian call, comparing five fixed cells (profile=smoke) with one continuous training trajectory per replication and checkpoints [0, 1, 2].

## Selected checkpoints by configuration
- w32_n100: checkpoint 2
- w16_n100: checkpoint 2
- w8_n100: checkpoint 2
- w32_n200: checkpoint 2
- w32_n400: checkpoint 2

## Sensitivity conclusions (descriptive)
- Whether reducing width improved held-out performance: w16_n100_vs_w32_n100 mean=0.0370518; w8_n100_vs_w32_n100 mean=0.0277097.
- Whether increasing training data improved held-out performance: w32_n200_vs_w32_n100 mean=-0.00873116; w32_n400_vs_w32_n100 mean=-0.099294.
- Whether the generalisation gap narrowed: see paired metric `paired_difference_log_generalization_gap` in capacity_data_paired_contrasts.csv.
- Consistency across replications: inspect 95% CIs and medians in capacity_data_paired_contrasts.csv.

## Epoch-200 extension flag
- extension_recommended=false

## Warnings and limitations
- None from automated validation.
- Training/validation/test/pilot are independent by construction and seeds are fixed deterministically.
- This is a sensitivity analysis rather than proof that parameter count alone caused the m=252 behaviour.
