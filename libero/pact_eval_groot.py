#!/usr/bin/env python
"""PACT-VLA on GR00T N1.7 / LIBERO.

flow_velocity (a replication of the official action-head forward at a given
(x_t, t)) and make_inputs come from groot_flow.py next to this file; the
policy itself is the unmodified Isaac GR00T N1.7 checkpoint.

Method (identical to the paper's LIBERO protocol): at each replan point the
policy sees the d-stale observation and draws K candidate chunks in one
batched get_action; the newest observation scores each candidate by the flow
residual  E_k = sum_t w_t || v(x_t, t | o_fresh) - (A_k - eps) ||^2  over the
live action dims, with one eps shared across the K candidates (CRN) and
gamma-decayed step weights; the argmin chunk executes for action_steps steps.
K=1 runs the same pipeline with no selection (the paired baseline).

GR00T's flow convention: x_t = (1-t)*noise + t*target,
velocity target = target - noise; t is the model's own bucketed time.
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import multiprocessing as mp

import numpy as np

STATE_KEYS = ["state.x", "state.y", "state.z", "state.roll", "state.pitch",
              "state.yaw", "state.gripper"]
VIDEO_KEYS = ["video.image", "video.wrist_image"]
LANG_KEY = "annotation.human.action.task_description"
ACTION_KEYS = ["action.x", "action.y", "action.z", "action.roll", "action.pitch",
               "action.yaw", "action.gripper"]


def obs_to_dict(obs):
    """Flat sim observation -> the {image, wrist_image, state, task} layout
    groot_flow.make_inputs consumes (8-dim state: 6 scalars + 2 gripper)."""
    return {
        "image": np.asarray(obs["video.image"], dtype=np.uint8),
        "wrist_image": np.asarray(obs["video.wrist_image"], dtype=np.uint8),
        "state": np.concatenate([np.atleast_1d(obs[k]) for k in STATE_KEYS]).astype(np.float32),
        "task": obs[LANG_KEY],
    }


class PactGroot:
    """K-candidate generation + fresh-observation flow-residual selection."""

    def __init__(self, wrapper, K, taus, gamma, log=False):
        import torch
        from groot_flow import make_inputs  # noqa: F401  (path set by caller)

        self.torch = torch
        self.wrapper = wrapper                     # Gr00tSimPolicyWrapper
        self.inner = wrapper.policy                # Gr00tPolicy
        self.model = self.inner.model
        self.head = self.model.action_head
        self.hd = next(self.head.parameters()).dtype
        self.K, self.taus, self.gamma, self.log = K, taus, gamma, log
        self.act_mask = None                       # resolved on first verify

    def _collated(self, obs_dict, k):
        """collate k replicas of one observation (full kwargs layout)."""
        from gr00t.data.types import MessageType
        from gr00t.policy.gr00t_policy import _rec_to_dtype
        from groot_flow import flat_to_nested

        flat = {
            "video.image": obs_dict["image"][None, None],
            "video.wrist_image": obs_dict["wrist_image"][None, None],
        }
        for i, key in enumerate(STATE_KEYS[:-1]):
            flat[key] = np.asarray([[obs_dict["state"][i]]], dtype=np.float32)[..., None]
        flat["state.gripper"] = np.asarray(obs_dict["state"][6:8], dtype=np.float32)[None, None]
        flat[LANG_KEY] = [obs_dict["task"]]
        # This stack mutates in place at several layers (process_backbone_output is the
        # documented one), so K aliased references to a single processed sample corrupt
        # each other under collation -- the K=8 random-pick control measurably undershot
        # the K=1 baseline (69.4 vs 73.0) with `[one] * k`. Build every replica through
        # its own processor pass instead.
        replicas = []
        states = None
        for _ in range(k):
            step = self.inner._to_vla_step_data(
                self.inner._unbatch_observation(flat_to_nested(self.inner, flat))[0])
            states = step.states
            replicas.append(self.inner.processor(
                [{"type": MessageType.EPISODE_STEP.value, "content": step}]))
        collated = self.inner.collate_fn(replicas)
        return _rec_to_dtype(collated, dtype=self.torch.bfloat16), states

    def candidates(self, stale_obs):
        """K chunks from the stale observation: one batched forward.
        Returns (normalized (K,T,D) tensor, list of K physical (T,7) chunks)."""
        torch = self.torch
        obs = obs_to_dict(stale_obs)
        collated, states = self._collated(obs, self.K)
        with torch.inference_mode():
            pred = self.inner.model.get_action(**collated)
        normalized = pred["action_pred"].float()                    # (K, T, D)
        batched_states = {k: np.stack([states[k]] * self.K, axis=0)
                          for k in self.inner.modality_configs["state"].modality_keys}
        physical = self.inner.processor.decode_action(
            normalized.cpu().numpy(), self.inner.embodiment_tag, batched_states)
        cols = []
        for k in ACTION_KEYS:
            v = np.asarray(physical[k.split("action.")[1]], dtype=np.float32)
            if v.ndim == 2:
                v = v[..., None]
            cols.append(v)
        phys = np.concatenate(cols, axis=-1)                        # (K, T, 7)
        return normalized, [phys[i] for i in range(self.K)]

    def select(self, fresh_obs, normalized):
        """argmin_k of the flow residual under the newest observation."""
        torch = self.torch
        from groot_flow import flow_velocity

        K, T, D = normalized.shape
        collated, _ = self._collated(obs_to_dict(fresh_obs), K)
        payload = collated.get("inputs", collated)
        bi, ai = self.model.prepare_input(payload)
        with torch.inference_mode():
            bo = self.model.backbone(bi)
            if self.act_mask is None:
                m = getattr(ai, "action_mask", None)
                if m is not None:
                    m = m.reshape(K, -1, D)[0, 0].bool().cpu()
                    self.act_mask = m
                else:                              # fall back to the 7 live dims
                    self.act_mask = torch.zeros(D, dtype=torch.bool)
                    self.act_mask[:7] = True
                if self.log:
                    print(f"[PACT] act_mask dims={int(self.act_mask.sum())}/{D}", flush=True)
            A = normalized.to(bo.backbone_features.device)
            eps = torch.randn(1, T, D, device=A.device).expand(K, T, D)   # CRN
            w = torch.tensor([self.gamma ** h for h in range(T)], device=A.device)
            w = w / w.sum()
            mask = self.act_mask.to(A.device)
            E = torch.zeros(K, device=A.device)
            u = A - eps
            for tv in self.taus:
                t = torch.full((K, 1, 1), float(tv), device=A.device)
                x_t = (1 - t) * eps + t * A
                v = flow_velocity(self.head, bo, ai, x_t.to(self.hd), t.to(self.hd)).float()
                res = ((v - u) ** 2)[..., mask].mean(dim=-1)              # (K, T)
                E = E + (res * w[None, :]).sum(dim=-1)
        idx = int(E.argmin().item())
        if self.log:
            gap = float((E.max() - E.min()) / (E.mean().abs() + 1e-9))
            print(f"[PACT] argmin={idx} relEgap={gap:.3f}", flush=True)
        return idx


def run_task(pact, task_bddl, task_desc, init_states, n_eps, delay, max_steps,
             horizon, action_steps, ep_offset=0):
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv

    env = LiberoEnv(task_bddl_file=task_bddl, task_description=task_desc)
    K_exec = action_steps if action_steps > 0 else max(1, delay)
    n_success = 0
    for ep in range(ep_offset, ep_offset + n_eps):
        obs, _ = env.reset()
        if init_states is not None:
            env._env.set_init_state(init_states[ep % len(init_states)])
            obs, _, _, _, _ = env.step({k: np.zeros(1) for k in ACTION_KEYS})
        buf = deque([obs] * (delay + 1), maxlen=delay + 1)
        chunk, idx = None, 0
        for _ in range(max_steps):
            if chunk is None or idx >= K_exec:
                normalized, phys = pact.candidates(buf[0])
                if pact.K <= 1:
                    sel = 0
                elif getattr(pact, "select_mode", "argmin") == "random":
                    sel = int(np.random.randint(pact.K))   # control: iid pick == K=1 in distribution
                else:
                    # verify=prev scores under o_{t+d-1} (newest frame that keeps verification
                    # off the commit deadline); default scores under the switch-time frame.
                    vm = getattr(pact, "verify_mode", "fresh")
                    if vm == "stale":
                        vobs = buf[0]          # generation frame o_t: mechanism control, no new information
                    elif vm == "prev" and len(buf) >= 2:
                        vobs = buf[-2]         # o_{t+d-1}: newest frame off the commit deadline
                    else:
                        vobs = buf[-1]         # o_{t+d}: switch-time frame
                    sel = pact.select(vobs, normalized)
                chunk, idx = phys[sel][:horizon], 0
            a = chunk[min(idx, len(chunk) - 1)]
            obs, _, done, _, info = env.step(
                {k: np.atleast_1d(a[i]) for i, k in enumerate(ACTION_KEYS)})
            buf.append(obs)
            idx += 1
            if info.get("success"):
                n_success += 1
                break
            if done:
                break
    env.close()
    return n_success


def make_units(n_tasks, per_task, nw, base_offset=0):
    """base_offset shifts the episode index each task starts from.

    NOTE: this does NOT change the initial states in the standard protocol. LIBERO
    ships exactly 50 initial states per task and a 500-episode suite run gives
    50 episodes per task, so `init_states[ep % 50]` already enumerates all of them;
    any offset that is a multiple of 50 replays the identical set in the identical
    order. Repeat runs therefore vary only through the policy's sampling noise,
    which is not seeded anywhere in this script. The flag is kept for partial runs
    (n_eps < 50 per task), where the offset does select a different subset."""
    chunks = max(1, min(per_task, -(-nw // n_tasks)))
    units = []
    for tid in range(n_tasks):
        base, extra = divmod(per_task, chunks)
        off = base_offset
        for c in range(chunks):
            n = base + (1 if c < extra else 0)
            if n:
                units.append((tid, off, n))
                off += n
    return units


def cpu_quota():
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    return os.cpu_count() or 4


def _worker(payload):
    (model_path, suite_name, units, delay, max_steps, horizon, threads,
     action_steps, k, taus, gamma, log, select_mode, verify_mode) = payload
    # Spawned children rebuild sys.path without the parent's script dir, and
    # groot_flow.py lives next to this file -- put it back explicitly.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    import torch
    torch.set_num_threads(threads)
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from libero.libero import benchmark

    wrapper = Gr00tSimPolicyWrapper(
        Gr00tPolicy(embodiment_tag=EmbodimentTag.LIBERO_PANDA,
                    model_path=model_path, device=0))
    pact = PactGroot(wrapper, k, taus, gamma, log=log)
    pact.select_mode = select_mode
    pact.verify_mode = verify_mode
    suite = benchmark.get_benchmark_dict()[suite_name]()
    out = []
    for tid, ep_off, n_eps in units:
        task = suite.get_task(tid)
        bddl = (suite.get_task_bddl_file_path(tid)
                if hasattr(suite, "get_task_bddl_file_path") else task.bddl_file)
        inits = suite.get_task_init_states(tid)
        n = run_task(pact, bddl, task.language, inits, n_eps, delay, max_steps,
                     horizon, action_steps, ep_offset=ep_off)
        out.append((tid, ep_off, n, n_eps))
        print(f"  [d={delay}] task {tid} eps {ep_off}+{n_eps} done ({n}/{n_eps})", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--delays", default="0")
    p.add_argument("--n-episodes", type=int, default=500)
    p.add_argument("--ep-offset", type=int, default=0,
                   help="shift the initial-state index; each LIBERO task has 50 initial states and "
                        "the standard 500-episode protocol already visits all of them, so offsets "
                        "that are multiples of 50 replay the same states")
    p.add_argument("--max-steps", type=int, default=720)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--action-steps", type=int, default=8)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--taus", default="0.1,0.3,0.5,0.7")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pact-log", type=int, default=0)
    p.add_argument("--select", default="argmin", choices=["argmin", "random"])
    p.add_argument("--verify", default="prev", choices=["fresh", "prev", "stale"])
    p.add_argument("--out", default="results/groot_pact.json")
    args = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = suite.n_tasks
    per_task = max(1, args.n_episodes // n_tasks)
    taus = [float(x) for x in args.taus.split(",")]
    print(f"suite={args.suite} tasks={n_tasks} eps/task={per_task} K={args.k} "
          f"taus={taus} gamma={args.gamma} action_steps={args.action_steps}", flush=True)

    nw = args.num_workers or max(1, min(n_tasks * per_task, max(1, cpu_quota() - 2), 10))
    results = {}
    for delay in [int(x) for x in args.delays.split(",")]:
        t0, total, done_eps = time.time(), 0, 0
        units = make_units(n_tasks, per_task, nw, base_offset=args.ep_offset)
        slices = [units[i::nw] for i in range(nw)]
        slices = [sl for sl in slices if sl]
        per_worker = max(1, cpu_quota() // max(1, len(slices)))
        payloads = [(args.model_path, args.suite, sl, delay, args.max_steps,
                     args.horizon, per_worker, args.action_steps, args.k, taus,
                     args.gamma, bool(args.pact_log), args.select, args.verify)
                    for sl in slices]
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            for chunk in pool.imap_unordered(_worker, payloads):
                for tid, ep_off, n, n_eps in chunk:
                    total += n
                    done_eps += n_eps
        sr = 100.0 * total / done_eps
        results[delay] = sr
        print(f"OVERALL {args.suite} K={args.k} d={delay}: SR={sr:.2f}% "
              f"({done_eps} eps, {time.time()-t0:.1f}s)", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
