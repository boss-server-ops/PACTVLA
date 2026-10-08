#!/bin/bash
# PACT-VLA / single-candidate baseline for pi_0.5 on LIBERO, run inside VLASH's batched evaluator.
#
# Required:
#   VLASH_ROOT   checkout of the official VLASH codebase
#   LIBERO_ROOT  checkout of LIBERO
#   CKPT         pi_0.5 LIBERO checkpoint directory
# Common knobs (defaults in brackets):
#   SUITE [libero_goal]  DELAY [4]  NEP [500]  NACT [5]  BATCH [10]  SEED [42]  OUT [results/<tag>]
#   FAST_PACT=K          enable selection over K candidates (omit for the single-candidate baseline)
#   PACT_VERIFY [prev]   scoring frame: prev = pre-commit o_{t+d-1}, fresh = o_{t+d}
#   FR_SLICE [NACT] / FR_GAMMA [1.0]  scored steps and their weights
#   FAST_NAIVE=1         delay the full observation, proprioception included (used for both arms)
#   THREADS=auto|<n>|0   per-process thread cap (auto: cgroup CPU quota / number of processes)
set -u
if [ -n "${D:-}" ] && [ -z "${DELAY:-}" ]; then echo "FATAL: pass DELAY=, not D="; exit 2; fi
: "${VLASH_ROOT:?set VLASH_ROOT}" "${LIBERO_ROOT:?set LIBERO_ROOT}" "${CKPT:?set CKPT}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export LIBERO_PATH=$LIBERO_ROOT
export PYTHONPATH=$HERE/libero:$VLASH_ROOT:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Inside containers os.cpu_count() reports the host's cores while the cgroup grants far fewer;
# numpy/torch/MuJoCo size their thread pools from cpu_count, so the main process plus BATCH
# environment workers oversubscribe badly. Cap threads at quota/processes, before any import.
if [ "${THREADS:-auto}" != "0" ]; then
  if [ "${THREADS:-auto}" = "auto" ]; then
    Q=$(awk '{if ($1=="max") print 0; else print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null)
    [ -z "$Q" ] && Q=$(awk -v p="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)" \
      '{if ($1>0 && p>0) print int($1/p); else print 0}' /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)
    NPROC=$(( ${BATCH:-10} + 1 ))
    THREADS=$(( ${Q:-8} > 0 ? ( ${Q:-8} + NPROC - 1 ) / NPROC : 1 )); [ "$THREADS" -lt 1 ] && THREADS=1
  fi
  export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
  export OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
fi

export FAST_NOISE=${FAST_NOISE:-0} FAST_NAIVE=${FAST_NAIVE:-1}
# Paper configuration for pi_0.5: score under o_{t+d-1}, over exactly the executed steps,
# uniformly weighted, at flow times 0.2/0.4/0.6/0.8.
export PACT_VERIFY=${PACT_VERIFY:-prev} FR_SLICE=${FR_SLICE:-${NACT:-5}} FR_GAMMA=${FR_GAMMA:-1.0}
export FR_TAUS=${FR_TAUS:-"0.2 0.4 0.6 0.8"} DELAY=${DELAY:-4}
D=${DELAY:-4}; SUITE=${SUITE:-libero_goal}; NEP=${NEP:-500}
TAG=${TAG:-${SUITE}_d${D}_k${FAST_PACT:-1}}
OUT=${OUT:-results/$TAG}; mkdir -p "$OUT"
echo "suite=$SUITE d=$D n=$NEP nact=${NACT:-5} K=${FAST_PACT:-1} verify=${PACT_VERIFY:-} -> $OUT"
$PY "$HERE/libero/eval_libero_noisy.py" \
  --policy.path="$CKPT" \
  --output_dir="$OUT" \
  --env.type=libero --env.task="$SUITE" \
  --eval.n_episodes="$NEP" --eval.batch_size=${BATCH:-10} --eval.use_async_envs=${ASYNC_ENVS:-True} \
  --policy.device=cuda --policy.use_amp=false --policy.n_action_steps=${NACT:-5} \
  --eval.async_delay="$D" --eval.method_type=vlash \
  --seed=${SEED:-42} --num_gpus=1
