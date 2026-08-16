# HANDOVER

## Stage 8: Frozen NCV Transfer, Calibration and Amortisation

### 1. Implementation facts

**New files added:**
- `asian_options/contracts.py` — Seven-contract grid (reference + 6 targets). `CONTRACT_IDS`, `TARGET_IDS`, `REFERENCE_ID`, `make_contract_cfg(contract_id, n_paths, seed)`, `validate_contract_grid()`. All contracts share S0=100, r=0.05, m=12; one parameter varies at a time from reference (K=100, sigma=0.20, T=1.0).
- `asian_options/frozen_transfer.py` — Frozen NCV transfer estimators. Key exports: `NearZeroVarianceError`, `compute_network_hash(network)`, `verify_frozen_hash(network, expected_hash)`, `train_reference_network(ref_cfg, n_training, train_seed)`, `ncv_transfer_beta1(...)`, `ncv_transfer_beta(...)`, `compute_high_precision_reference(...)`. Parameter hash is SHA-256 over W1,b1,W2,b2 byte arrays; verified before and after each target evaluation. `NearZeroVarianceError` is raised explicitly when `Var(C0) < 1e-12` or non-finite; no silent fallback.
- `scripts/run_stage8.py` — Stage 8 experiment runner. Profiles: `smoke` (n_train=100, n_pilot=50, n_pricing=200, 2 reps) and `dissertation` (n_train=5000, n_pilot=1000, n_pricing=50000, 30 reps). Implements all required output files, seed scheduling, amortised cost analysis, break-even computation, and validation report.
- `asian_options/tests/test_stage8.py` — 47 tests covering: contract grid, one-parameter-change rule, monitoring dates, network hash, NearZeroVarianceError, amortised costs, break-even formula, beta=1 invariant, estimator variance identity, AV pair accounting, seed independence, validation report, config snapshot, analytical expectation, result keys, and no-regression checks.

**Architecture preserved:**
- Shallow one-hidden-layer ReLU network (`_ShallowNet`) from `neural_cv.py` — unchanged.
- `analytical_network_expectation` from `neural_cv.py` — reused directly.
- All existing estimators (`standard_monte_carlo`, `antithetic_variates`, `geometric_control_variate`) — unchanged.
- All existing tests continue to pass (same 12 torch-related failures as before, now requiring `pip install torch`).

### 2. Validation facts

- Contract grid: `validate_contract_grid()` tests one-parameter-change rule, dt=T/m correctness, and complete key set.
- Frozen hash: SHA-256 verified before and after every target evaluation in `ncv_transfer_beta1` and `ncv_transfer_beta`.
- Near-zero variance: `NearZeroVarianceError` raised with diagnostics (contract params, n_pilot, Var(C0) value).
- Seed independence: distinct stream seeds per contract/phase/replication verified in tests.
- AV pair accounting: pricing_simulated_paths == 2 * pricing_observations enforced in validation report.
- Estimator variance identity: est_var = obs_var / n_pricing checked per row.

### 3. Empirical results

NCV-based methods (NCV_SCRATCH, NCV_TRANSFER_BETA1, NCV_TRANSFER_BETA) require PyTorch. In the current environment (no torch), these methods produce `error='torch_not_available'` rows. All non-NCV estimators (MC, AV, GCV) run successfully for all 7 contracts.

### 4. Cautious interpretations

- Break-even analysis uses mean timing estimates and may have high variance with few replications.
- Transfer effectiveness depends on correlation between reference H0(Z) and target payoffs; contracts with very different dynamics (e.g., volatility_high) may show reduced benefit.

### 5. Limitations / risks

- Torch not installed in CI environment; NCV methods produce error rows.
- Dissertation profile requires torch and ~30 replications of compute time.

### Commands

```bash
# Smoke test (all contracts, all methods, 2 reps, tiny sizes)
python scripts/run_stage8.py --profile smoke --output-dir experiment_runs

# Dissertation profile (requires torch, 30 reps)
python scripts/run_stage8.py --profile dissertation --output-dir experiment_runs

# Run Stage 8 tests
pytest asian_options/tests/test_stage8.py -v

# Run full test suite
pytest -q
```

### Seed schedule

Per replication, seed streams are derived as:
  `seed = base_seed + replication * 100_000 + OFFSET`
with offsets: ref_train=1000, ref_val=2000, target_train=3000+ci*100,
pilot=4000+ci*100, pricing=5000+ci*100, high_prec=9000.

All realized seeds are saved in `seed_manifest.csv` in each output bundle.

## Stage 8 + training-curve accounting update (current)

- Stage 8 fixed NCV epoch is now 25 with
  `ncv_epoch_source=training_curve_validation_tuning`.
- Training objective remains MSE; checkpoint selection criterion is validation
  residual variance from the separate training-curve tuning workflow.
- The recorded tuning evidence is retained as interpretation context: roughly
  26% held-out residual-variance improvement at 25 epochs versus 100 epochs,
  while GCV remains materially more variance-efficient (about 5.8–5.9× better
  residual variance than NCV).
- End-to-end pricing runtime fields are recorded for cost projection
  (path/payoff, control evaluation, estimator reduction, and full pricing
  runtime), and projected total cost uses:
  `setup_cost + Q * marginal_pricing_cost`.
- Validation report generation now performs a post-write self-check so
  `training_curve_validation_report.json` no longer fails because it checked for
  itself before creation.
- No architecture changes, no option-dependent rescaling estimator, and no
  non-GBM extensions were added.
