"""PACT-VLA inside VLASH's batched LIBERO evaluator.

The method needs no simulator access. It samples K chunks from the observation it already has,
executes the committed chunk through the delay window, and at the planned switch scores those K
chunks against the pre-commit observation that has arrived by then. That is pure policy
compute, so it runs inside the official batched harness: no environment forking, no simulator
state restore, and no privileged access.

Shapes: B environments in the batch, K candidates each. Generation and verification both fold
K into the batch dimension, so each costs one forward at batch B*K.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from vlash.policies.pi05.modeling_pi05 import build_attention_mask_and_position_ids

FR_TAUS = [float(x) for x in os.environ.get("FR_TAUS", "0.2 0.4 0.6 0.8").split()]
FR_GAMMA = float(os.environ.get("FR_GAMMA", "0.5"))
# The policy predicts a 50-step chunk but only the first n_action_steps execute. FR_SLICE
# restricts the residual to the first N steps (0 = whole chunk): with gamma=0.5 the score
# effectively sees only steps 0-4, yet under whole-chunk commitment the candidates' outcome
# divergence lives in the later executed steps. FR_SLICE=8 + FR_GAMMA=1 scores exactly the
# executed segment, uniformly.
FR_SLICE = int(os.environ.get("FR_SLICE", "0"))
# Speculative state hypotheses: candidate 0 conditions on the reported proprio, candidates 1..K-1 on proprio
# perturbed by sqrt(d)-scaled execution noise -- the emulation of K integration hypotheses of
# the boundary state. 0 disables (plain sampling).
PACT_HYP = float(os.environ.get("PACT_HYP", "0"))
# Candidate-diversity temperature: scale the initial flow noise of candidates 1..K-1 by this
# factor (candidate 0 stays at 1.0, so the plain sample is always in the pool). The pool on
# goal is outcome-homogeneous; this widens it while the selector filters the junk.
PACT_TEMP = float(os.environ.get("PACT_TEMP", "1"))
# Acceptance margin (speculative-decoding style): commit the argmin candidate only when its
# relative advantage over candidate 0 -- the sample that would execute with no selection --
# exceeds this threshold. At zero delay-damage the between-candidate E gaps are noise, and
# unconditional argmin trades the default sample for variance (the d1 tax); with a margin the
# selection engages exactly when the evidence is decisive. 0 = plain argmin.
PACT_MARGIN = float(os.environ.get("PACT_MARGIN", "0"))
PACT_DELAY = int(os.environ.get("DELAY", os.environ.get("PACT_DELAY", "0")) or 0)
PACT_LOG = os.environ.get("PACT_LOG", "0") == "1"

# Selector family. "flowres" (default) = this paper's score: the policy's own flow residual under
# the pre-commit observation. "keystone" = KeyStone (Geometry Guided Self-Consistency, arXiv
# 2605.08638, Apache-2.0, vendored under scripts/third_party/keystone): geometry-only consensus --
# cluster the K candidates and commit the medoid of the dominant mode. It never looks at an
# observation, so it costs no forward pass and is delay-agnostic by construction; that contrast is
# exactly what the comparison is for. KS_* mirror the authors' pi0.5-LIBERO config
# (cluster_medoid_guarded, C=2, tau=0.3, l2).
PACT_SELECT = os.environ.get("PACT_SELECT", "flowres").lower()
KS_C = int(os.environ.get("KS_C", "2"))
KS_TAU = float(os.environ.get("KS_TAU", "0.3"))
KS_DIST = os.environ.get("KS_DIST", "l2")


def _repeat(obs: dict, K: int) -> dict:
    """Tile every environment's observation K times, keeping env-major order."""
    out = {}
    for k, v in obs.items():
        if k == "task":
            out[k] = [t for t in v for _ in range(K)]
        elif torch.is_tensor(v):
            out[k] = v.repeat_interleave(K, dim=0)
        else:
            out[k] = v
    return out


class PactController:
    def __init__(self, policy, K: int, h_exec: int, verify: str = "fresh"):
        self.policy, self.K, self.H = policy, K, h_exec
        self.verify = verify                      # fresh = o_t (what has arrived), stale = o_{t-d}
        self.orig = policy.config.action_feature.shape[0]

    def _prep(self, batch):
        b = self.policy.normalize_inputs(batch)
        images, img_masks = self.policy.prepare_images(b)
        state = self.policy.prepare_state(b)
        lt, lm = self.policy.prepare_language(b, pad_to_max_length=False)
        return images, img_masks, lt, lm, state

    def candidates(self, obs: dict):
        """K chunks per environment from the observation the policy is allowed to see.

        Diversity is the flow-matching sampling noise, which is the only source the deployable
        method has: one call at batch B*K, so K candidates cost one generation, not K.
        """
        B = len(obs["task"])
        with torch.inference_mode():
            images, img_masks, lt, lm, state = self._prep(_repeat(obs, self.K))
            if PACT_HYP > 0 and self.K > 1:
                pert = torch.randn_like(state) * (PACT_HYP * max(PACT_DELAY, 1) ** 0.5)
                keep = (torch.arange(state.shape[0], device=state.device) % self.K == 0)
                state = torch.where(keep[:, None], state, state + pert)
            noise = None
            if PACT_TEMP != 1.0 and self.K > 1:
                cfg = self.policy.config
                noise = self.policy.model.sample_noise(
                    (state.shape[0], cfg.chunk_size, cfg.max_action_dim), state.device)
                keep = (torch.arange(state.shape[0], device=state.device) % self.K == 0)
                noise = torch.where(keep[:, None, None], noise, noise * PACT_TEMP)
            full = self.policy.model.sample_actions(images, img_masks, lt, lm, state, noise=noise)
        chunk, maxdim = full.shape[1], full.shape[2]
        exec_c = self.policy.unnormalize_outputs({"action": full[:, :, : self.orig]})["action"]
        exec_c = exec_c[:, : self.H].float().cpu().numpy().reshape(B, self.K, self.H, self.orig)
        return full.detach().reshape(B, self.K, chunk, maxdim), exec_c

    def _select_keystone(self, full: torch.Tensor) -> np.ndarray:
        """KeyStone's guarded cluster-medoid over the same candidate pool.

        Fidelity notes, all load-bearing: the authors select on the raw model output *before*
        unnormalisation (their LeRobot template hooks the sampler), restrict the distance to the
        real action dims so zero padding cannot dominate, and truncate to the executed steps.
        Their block layout is (K, B, T, D) while ours is env-major (B, K, ...) -- mismatching the
        two silently mixes candidates across environments, so the permute below is required.
        The selector returns a real candidate chunk, never an interpolation, which is what lets
        us recover the index our harness needs.
        """
        import sys, os as _os
        _tp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "third_party")
        if _tp not in sys.path:
            sys.path.insert(0, _tp)
        from keystone import cluster_medoid_select

        B, K = full.shape[0], full.shape[1]
        cand = full.permute(1, 0, 2, 3).float()                      # (B,K,...) -> (K,B,T,D)
        sel = cluster_medoid_select(cand, C=KS_C, tau=KS_TAU, distance=KS_DIST,
                                    action_dim=self.orig, executed_steps=self.H)
        ref = cand[:, :, : sel.shape[1], : sel.shape[2]]              # (K,B,H,D)
        hit = (sel.unsqueeze(0) == ref).all(dim=-1).all(dim=-1)       # (K,B)
        idx = hit.float().argmax(dim=0).cpu().numpy()
        if PACT_LOG:
            print(f"[KEYSTONE] pick={idx.tolist()} unique={[int(h.sum()) for h in hit.T.cpu()]}",
                  flush=True)
        return idx

    def select(self, obs: dict, full: torch.Tensor) -> np.ndarray:
        """argmin_k of the policy's own flow-matching residual under `obs`.

        Common random numbers: one epsilon per environment, shared by that environment's K
        candidates, so the comparison between candidates is paired.
        """
        if PACT_SELECT == "keystone":
            return self._select_keystone(full)
        if PACT_SELECT == "random":
            # Mechanism control: same pool, same commitment, no information used. An i.i.d. pick
            # from K i.i.d. candidates is distributionally the single-candidate baseline, so any
            # gap between this and FlowRes is attributable to the score, not to sampling K chunks.
            return np.random.randint(full.shape[1], size=full.shape[0])
        B, K, chunk, maxdim = full.shape
        m = self.policy.model
        with torch.inference_mode():
            images, img_masks, lt, lm, state = self._prep(_repeat(obs, K))
            prefix_embs, ppm, pam = m.prefix_embedder(images, img_masks, lt, lm)
            for layer in m.layers:
                layer.self_attn.attn.reset_cache()
            att, pos = build_attention_mask_and_position_ids(ppm, pam, prefix_embs.dtype)
            hs = [prefix_embs, None]
            for layer in m.layers:
                hs = layer(hs, att, pos, [None, None], use_cache=True)   # one prefill at B*K
            cand = full.reshape(B * K, chunk, maxdim)
            eps = m.sample_noise((B, 1, chunk, maxdim), cand.device).expand(B, K, chunk, maxdim)
            eps = eps.reshape(B * K, chunk, maxdim)
            pw = torch.tensor([FR_GAMMA ** h for h in range(chunk)], device=cand.device)
            if FR_SLICE > 0:
                pw[FR_SLICE:] = 0.0
            pw = pw / pw.sum()
            u = eps - cand
            E = torch.zeros(B * K, device=cand.device)
            for tau in FR_TAUS:
                x_t = tau * eps + (1.0 - tau) * cand
                tt = torch.full((B * K,), float(tau), device=cand.device)
                v = m.denoise_step(ppm, pam, state, x_t, tt)
                res = ((v - u) ** 2)[..., : self.orig].mean(dim=-1)       # [B*K, chunk]
                E = E + (res * pw[None, :]).sum(dim=-1)
        Er = E.reshape(B, K)
        idx_t = Er.argmin(dim=1)
        if PACT_MARGIN > 0:
            adv = (Er[:, 0] - Er.gather(1, idx_t[:, None])[:, 0]) / (Er.mean(dim=1).abs() + 1e-9)
            idx_t = torch.where(adv < PACT_MARGIN, torch.zeros_like(idx_t), idx_t)
        idx = idx_t.cpu().numpy()
        if PACT_LOG:
            gap = (Er.max(dim=1).values - Er.min(dim=1).values) / (Er.mean(dim=1).abs() + 1e-9)
            print(f"[PACT] argmin={idx.tolist()} relEgap={[round(float(g),3) for g in gap]}",
                  flush=True)
        return idx
