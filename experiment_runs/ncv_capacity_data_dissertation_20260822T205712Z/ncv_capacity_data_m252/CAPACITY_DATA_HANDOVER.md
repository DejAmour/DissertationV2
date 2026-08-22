# NCV Capacity-vs-Data Sensitivity Handover

## Exact experiment completed
Additive sensitivity run for m=252 on the reference arithmetic Asian call, comparing five fixed cells (profile=dissertation) with one continuous training trajectory per replication and checkpoints [0, 10, 25, 50, 100, 200].

## Selected checkpoints by configuration
- w32_n5000: checkpoint 25
- w16_n5000: checkpoint 25
- w8_n5000: checkpoint 25
- w32_n10000: checkpoint 200
- w32_n20000: checkpoint 200

## Sensitivity conclusions (descriptive)
- Whether reducing width improved held-out performance: w16 vs w32 log residual-variance ratio mean=-0.18399; w8 vs w32 mean=-0.308316.
- Whether increasing training data improved held-out performance: w32 n10000 vs n5000 mean=-0.955172; w32 n20000 vs n5000 mean=-2.68145.
- Whether the generalisation gap narrowed: see paired metric `paired_difference_log_generalization_gap` in capacity_data_paired_contrasts.csv.
- Consistency across replications: inspect 95% CIs and medians in capacity_data_paired_contrasts.csv.

## Epoch-200 extension flag
- extension_recommended=true

## Warnings and limitations
- extension_recommended_true
- Training/validation/test/pilot are independent by construction and seeds are fixed deterministically.
- This is a sensitivity analysis rather than proof that parameter count alone caused the m=252 behaviour.
