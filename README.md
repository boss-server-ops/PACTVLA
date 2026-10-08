# PACT-VLA: Speculative Action-Chunk Verification with Pre-Commit Observations

PACT-VLA is a training-free test-time method for flow-based vision-language-action (VLA)
policies executed under inference delay. While a new action chunk is being generated, the
robot keeps acting and the scene keeps changing, so the chunk may no longer fit the scene by
the time it is executed. PACT-VLA generates a fixed pool of $N$ candidate chunks from the
observation available at inference start, then, just before the planned chunk switch at
$t+d$, scores every candidate with **FlowRes** under the *pre-commit observation*
$o_{t+d-1}$ and commits the lowest-scoring candidate in full.

FlowRes reuses the frozen flow policy itself: for each candidate it builds flow-time inputs
from one shared noise sample, queries the policy's velocity under the pre-commit observation,
and accumulates the squared velocity residuals. No additional training, verifier, world
model, or simulator access is required.

This repository contains the LIBERO evaluation code for two backbones, $\pi_{0.5}$ and
GR00T N1.7.

## Repository structure

```
libero/
  pact_fast.py           batched candidate generation and FlowRes selection for pi_0.5
  eval_libero_noisy.py   VLASH's batched LIBERO evaluator with delay emulation and the PACT-VLA hook
  pact_eval_groot.py     standalone LIBERO evaluator for GR00T N1.7 with PACT-VLA
  groot_flow.py          GR00T helpers: velocity at a given (x_t, t), observation collation
scripts/
  run_libero_pi05.sh     pi_0.5 entry point (environment-variable driven)
  run_libero_groot.sh    GR00T N1.7 entry point
```

## Installation

**pi_0.5.** The official [VLASH](https://github.com/vlash-project) codebase together with
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and a $\pi_{0.5}$ LIBERO
checkpoint. `eval_libero_noisy.py` is run in place of VLASH's evaluator; it imports
`pact_fast.py` from the same directory.

**GR00T N1.7.** The [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) codebase with the
per-suite LIBERO checkpoints. The GR00T policy code is used unmodified.

No component of this repository performs training.

## Evaluation

$\pi_{0.5}$, PACT-VLA with $N=8$ at delay $d=4$ on one suite:

```bash
VLASH_ROOT=<vlash> LIBERO_ROOT=<libero> CKPT=<pi05_libero_checkpoint> \
SUITE=libero_goal DELAY=4 FAST_PACT=8 bash scripts/run_libero_pi05.sh
```

Omit `FAST_PACT` for the single-candidate baseline. The script defaults to the reported
configuration: 5 executed actions per chunk, scoring under $o_{t+d-1}$ over exactly the
executed steps with uniform weights, flow times $\{0.2,0.4,0.6,0.8\}$, 500 episodes.
`FAST_NAIVE=1` (default) delays the full observation, proprioception included, for both arms.

GR00T N1.7:

```bash
GROOT_ROOT=<isaac_groot> CKPT=<groot_suite_checkpoint> \
SUITE=libero_object K=8 bash scripts/run_libero_groot.sh
```

`K=1` gives the single-candidate baseline through the same path. `VERIFY` chooses the scoring
frame among `stale` ($o_t$), `prev` ($o_{t+d-1}$, default) and `fresh` ($o_{t+d}$), and
`SELECT=random` replaces the score with a uniform random pick.

## Notes

- A single noise sample is shared across the $N$ candidates and all verification flow times
  (common random numbers), so candidate scores are compared on the same interpolation path.
- The GR00T evaluator does not seed the policy's sampling noise. Runs of different arms are
  matched on task and initial state, not paired decision by decision.
- Inside containers, `os.cpu_count()` often reports the host's cores. The $\pi_{0.5}$ script
  caps per-process threads from the cgroup CPU quota; set `THREADS=<n>` to override.

## Acknowledgements

The $\pi_{0.5}$ evaluation builds on the official VLASH codebase and the LIBERO benchmark; the
GR00T evaluation uses Isaac GR00T. We thank the authors for releasing their code.
