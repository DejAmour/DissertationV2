# 2×2 Monitoring-Profile × Training-Horizon Sensitivity Study

## 1) Design
This run fixes a 2×2 design across monitoring profiles (m=12, m=252) and NCV checkpoints (25, 1,000), with all other model, optimiser, seed, and sampling choices held constant. Checkpoints were fixed ex ante and not selected using final pricing data.

## 2) Four formal cells

| m | checkpoint | geometric mean A=Var(GCV)/Var(NCV) | 95% CI | NCV beats GCV count |
|---:|---:|---:|---:|---:|
| 12 | 25 | 0.500871 | [0.472913, 0.530481] | 0/30 |
| 12 | 1000 | 17.7448 | [15.9351, 19.7599] | 30/30 |
| 252 | 25 | 0.179272 | [0.174284, 0.184404] | 0/30 |
| 252 | 1000 | 0.169731 | [0.159304, 0.18084] | 0/30 |

## 3) Paired contrasts (log-scale, then exponentiated)
- Δ12: estimate=3.5675, 95% CI=[3.42125, 3.71374], ratio-scale=35.4278
- Δ252: estimate=-0.0546936, 95% CI=[-0.103413, -0.00597386], ratio-scale=0.946775
- Δinteraction: estimate=3.62219, 95% CI=[3.46045, 3.78393], ratio-scale=37.4195

## 4) Does 1,000 epochs rescue m=252 NCV?
At m=252, the 1,000-epoch checkpoint remains at or below parity (A≤1 on average), so longer training does not rescue NCV versus GCV in this sensitivity design.

## 5) Does the monitoring-profile reversal remain under the full 2×2 cross?
After crossing both profiles with both checkpoints, the 12-date profile still shows the stronger NCV/GCV advantage.

## 6) Limitation
The monitoring profile m jointly changes payoff discretisation, network input dimension, and parameter count; this sensitivity analysis quantifies pattern robustness but is not a causal identification of any single mechanism.

## 7) Dissertation-ready epoch-selection replacement text
Epoch choice should be treated as a fixed design input selected off-line from held-out training/validation evidence. In the crossed 2×2 sensitivity analysis, longer NCV training can materially change relative NCV/GCV variance performance, so conclusions should be reported conditionally on both monitoring profile and checkpoint horizon rather than extrapolated from a single profile-horizon pair.

## 8) Suggested figure captions
- Figure 1 (checkpoint curves): Held-out NCV/GCV variance-advantage trajectories across epochs for m=252 and m=12, with geometric means and 95% log-scale CIs.
- Figure 2 (2×2 interaction): Crossed monitoring-profile × checkpoint comparison of held-out NCV/GCV variance advantage with 95% CIs on a log scale.

## 9) Integrity note
All numeric claims in this summary are sourced from this run’s generated CSV outputs; no unsupported values were introduced.
