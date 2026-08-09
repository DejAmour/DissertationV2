# Asian-options Stage 1 environment metadata

## User-validated dependency baseline

- Platform: Windows PowerShell workflow (`.\.venv\Scripts\Activate.ps1`)
- Python: 3.13.1
- numpy: 2.5.2
- scipy: 1.18.0
- torch: 2.13.0+cpu
- pytest: 9.1.1
- Deterministic PyTorch operations enabled: yes, via `asian_options.seed_everything()`

## Audit run captured in this repository update

- Platform: Linux-6.17.0-1020-azure-x86_64-with-glibc2.39
- Python: 3.12.3
- numpy: 2.5.2
- scipy: 1.18.0
- torch: 2.13.0+cu130
- pytest: 9.1.1
- CUDA available: no
- Deterministic PyTorch operations enabled: yes

## Determinism notes

- `seed_everything()` seeds Python `random`, NumPy, PyTorch, and CUDA when available.
- PyTorch deterministic algorithms are requested where practical.
- Results can still vary across Python versions, wheel builds, hardware, and drivers.
