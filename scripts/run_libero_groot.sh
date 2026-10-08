#!/bin/bash
# PACT-VLA / single-candidate baseline for GR00T N1.7 on LIBERO (official protocol:
# 8 executed actions per chunk, 720 max steps, 500 episodes per suite).
#
# Required:
#   GROOT_ROOT  checkout of Isaac GR00T (provides gr00t.* and the LIBERO env wrapper)
#   CKPT        GR00T N1.7 LIBERO checkpoint for SUITE
# Knobs [defaults]: SUITE [libero_object]  DELAYS [1,2,3,4]  K [8]  VERIFY [prev]  SELECT [argmin]
#   K=1 gives the single-candidate baseline through the same code path.
#   VERIFY selects the scoring frame: stale = o_t, prev = o_{t+d-1}, fresh = o_{t+d}.
set -u
: "${GROOT_ROOT:?set GROOT_ROOT}" "${CKPT:?set CKPT}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$HERE/libero:$GROOT_ROOT${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL=${MUJOCO_GL:-egl}
SUITE=${SUITE:-libero_object}; K=${K:-8}
${PYTHON:-python} "$HERE/libero/pact_eval_groot.py" --model-path "$CKPT" \
  --suite "$SUITE" --delays "${DELAYS:-1,2,3,4}" --n-episodes ${NEP:-500} \
  --action-steps 8 --max-steps 720 --k "$K" \
  --verify "${VERIFY:-prev}" --select "${SELECT:-argmin}" --taus 0.1,0.3,0.5,0.7 --gamma 0.5 \
  --out "${OUT:-results/${SUITE}_k${K}.json}"
