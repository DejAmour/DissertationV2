from __future__ import annotations

from asian_options.ncv_capacity_data_missing_cells_m252 import _full_grid_cells, _summary_table


def test_summary_table_emits_nine_unique_cells():
    cells = _full_grid_cells()
    selected = {c.config_id: 10 for c in cells}
    rows = []
    for rep in range(10):
        for c in cells:
            rows.append(
                {
                    "config_id": c.config_id,
                    "replication": rep,
                    "checkpoint": 10,
                    "split": "test",
                    "centered_residual_variance": 1.0 + rep / 100.0,
                    "ncv_vrr_vs_mc": 2.0 + rep / 100.0,
                    "payoff_network_correlation": 0.5,
                    "generalization_gap_log": 0.1,
                    "ncv_setup_cost_s": 3.0,
                }
            )
    summary = _summary_table(rows, selected, cells)
    assert len(summary) == 9
    assert len({row["config_id"] for row in summary}) == 9

