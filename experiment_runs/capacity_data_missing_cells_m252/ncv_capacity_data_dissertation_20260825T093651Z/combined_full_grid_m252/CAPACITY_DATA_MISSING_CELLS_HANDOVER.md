# NCV Capacity Data Missing Cells (m=252)

## Final questions
- 1) Increasing data for widths 8 and 16: assess from within-width residual-variance declines across 5k→10k→20k in combined table.
- 2) At 10,000 paths, best-performing configuration among the nine width–training-size combinations examined: w16_n10000.
- 3) At 20,000 paths, best-performing configuration among the nine width–training-size combinations examined: w16_n20000.
- 4) Width-32 at 20,000 remains best at 20,000 paths: no.
- 5) Association pattern: compare train-size trends within width and cross-width comparisons at fixed data using paired_contrasts.
- 6) Any new config outperforming existing w32_n20000: yes.
- 7) Any new config outperforming GCV benchmark (test residual variance): yes.
- 8) Consistency across 10 replications: use paired-contrast CI signs and medians in combined paired-contrast output.
- 9) Selected checkpoint at edge (200) for new cells: w8_n10000, w8_n20000, w16_n10000, w16_n20000.