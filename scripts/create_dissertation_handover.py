#!/usr/bin/env python3
"""
Create a comprehensive handover zip file containing all principal results,
configurations, and scripts from the DissertationV2 repository.

Usage:
    python scripts/create_dissertation_handover.py [--output-path OUTPUT_PATH]

The script organizes the following materials into a single zip:
  - Principal results (m=12 and m=252 metrics by contract)
  - Equal-observation and equal-path-budget outputs
  - Frozen-reuse results (β=1 and estimated β)
  - Runtime and break-even outputs
  - Reference-price calculations
  - Configuration files and timing/break-even scripts
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path
from datetime import datetime


def get_repo_root():
    """Find the repository root directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / '.git').exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root")


def create_handover_zip(output_path=None):
    """
    Create a comprehensive handover zip file.
    
    Args:
        output_path: Path where the zip file should be created. 
                    Defaults to <repo_root>/dissertation_handover.zip
    """
    repo_root = get_repo_root()
    
    if output_path is None:
        output_path = repo_root / 'dissertation_handover.zip'
    else:
        output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define files and folders to include
    files_to_include = [
        # Principal results
        ('experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/replication_level_results.csv', 
         'results/principal_results/replication_level_results.csv'),
        ('experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/cell_summary.csv', 
         'results/principal_results/cell_summary.csv'),
        ('experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/paired_contrasts.csv', 
         'results/principal_results/paired_contrasts.csv'),
        
        # Equal-observation and equal-path-budget outputs
        ('asian_options_equal_obs_comparison.csv', 
         'results/equal_observation_budget/equal_obs_comparison.csv'),
        ('asian_options_equal_budget_comparison.csv', 
         'results/equal_observation_budget/equal_budget_comparison.csv'),
        
        # Runtime and break-even outputs
        ('asian_options_runtime_comparison.csv', 
         'results/runtime_breakeven/runtime_comparison.csv'),
        ('experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/runtime_results.csv', 
         'results/runtime_breakeven/runtime_results.csv'),
        ('experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/checkpoint_curve_results.csv', 
         'results/runtime_breakeven/checkpoint_curve_results.csv'),
        
        # Reference-price and validation
        ('experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/dissertation_summary.md', 
         'results/validation/dissertation_summary.md'),
        ('experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/validation_report.json', 
         'results/validation/validation_report.json'),
        
        # Configuration files
        ('experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/config.json', 
         'configs/experiment_config.json'),
        ('experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/environment.json', 
         'configs/environment.json'),
        ('asian_options/config.py', 
         'configs/asian_options_config.py'),
        
        # Scripts for timing and break-even calculations
        ('scripts/run_stage8.py', 
         'scripts/run_stage8.py'),
        ('scripts/run_experiments.py', 
         'scripts/run_experiments.py'),
        ('scripts/run_stage8_sensitivity_2x2.py', 
         'scripts/run_stage8_sensitivity_2x2.py'),
        ('asian_options/results.py', 
         'scripts/asian_options_results.py'),
        ('asian_options/frozen_transfer.py', 
         'scripts/asian_options_frozen_transfer.py'),
        
        # Seed manifest and metadata
        ('experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/seed_manifest.csv', 
         'metadata/seed_manifest.csv'),
        ('asian_options/environment_metadata.md', 
         'metadata/environment_metadata.md'),
        
        # Documentation
        ('HANDOVER.md', 'README.md'),
    ]
    
    # Create zip file
    print(f"Creating handover zip: {output_path}")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for source_rel, dest_in_zip in files_to_include:
            source_path = repo_root / source_rel
            
            if not source_path.exists():
                print(f"⚠️  Warning: File not found: {source_rel}")
                continue
            
            # Add file to zip with desired path
            zf.write(source_path, arcname=dest_in_zip)
            print(f"✓ Added: {dest_in_zip}")
        
        # Add metadata file to zip root
        metadata_content = f"""# Dissertation Handover Package
Generated: {datetime.now().isoformat()}
Repository: DejAmour/DissertationV2

## Package Contents

### Results/
- **principal_results/**: Main results by contract (m=12, m=252)
  - replication_level_results.csv
  - cell_summary.csv
  - paired_contrasts.csv

- **equal_observation_budget/**: Equal-observation and equal-path-budget comparisons
  - equal_obs_comparison.csv
  - equal_budget_comparison.csv

- **runtime_breakeven/**: Runtime and break-even analysis
  - runtime_comparison.csv
  - runtime_results.csv
  - checkpoint_curve_results.csv

- **validation/**: Reference-price calculations and validation reports
  - dissertation_summary.md
  - validation_report.json

### Configs/
- experiment_config.json
- environment.json
- asian_options_config.py

### Scripts/
- run_stage8.py (main experiment runner with timing/break-even)
- run_experiments.py
- run_stage8_sensitivity_2x2.py
- asian_options_results.py (results processing)
- asian_options_frozen_transfer.py (frozen-reuse for β=1 and estimated β)

### Metadata/
- seed_manifest.csv (all random seeds used)
- environment_metadata.md

### README.md
- HANDOVER documentation with full implementation details

## No Further Testing Required
All results are pre-computed and ready for analysis. No additional experiments need to be run.
"""
        zf.writestr('PACKAGE_INFO.md', metadata_content)
        print(f"✓ Added: PACKAGE_INFO.md")
    
    print(f"\n✅ Handover zip created successfully!")
    print(f"   Location: {output_path}")
    print(f"   Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create comprehensive dissertation handover zip package'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Path where zip file should be created (default: <repo_root>/dissertation_handover.zip)'
    )
    
    args = parser.parse_args()
    
    try:
        zip_path = create_handover_zip(args.output_path)
        print(f"\n📦 Package ready at: {zip_path}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error creating handover zip: {e}", file=sys.stderr)
        sys.exit(1)
