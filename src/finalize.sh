#!/bin/bash
# Final pipeline: policy comparison + per-slot round-by-round plans.
set -e
cd /home/user/fantasy
python3 -W ignore src/experiment.py 700 2>&1 | tee output/experiment_v2.log
python3 -W ignore src/slot_plan.py 700 2>&1 | tail -2
echo "FINALIZE_COMPLETE"
