"""Run every step of the REV6 paper analysis in order (see README.md)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_verify_indomain.py", "02_verify_transfer.py", "03_tables.py", "04_episodes.py",
         "05_robustness.py", "06_figures.py"]
start = int(sys.argv[1]) if len(sys.argv) > 1 else 1        # e.g. "python run_all.py 3" resumes at step 3
for s in STEPS[start - 1:]:
    print(f"\n===== {s} =====", flush=True)
    subprocess.run([sys.executable, str(HERE / s)], check=True, cwd=HERE)
print("\nAll steps finished. Outputs are in paper_outputs_rev6/.")
