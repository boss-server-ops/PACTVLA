"""Velocity and input helpers for scoring candidates with GR00T N1.7.

The official action head only exposes forward(), which samples its own flow
time and noise. FlowRes needs the predicted velocity at a given (x_t, t), so
flow_velocity replicates that forward pass for caller-supplied inputs.
make_inputs turns raw LIBERO observations into the collated model inputs.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

STATE_KEYS = ["state.x", "state.y", "state.z", "state.roll", "state.pitch",
              "state.yaw", "state.gripper"]


def flat_to_nested(policy, flat):
    """Flat sim-format observation -> the nested layout Gr00tPolicy expects.

    Mirrors Gr00tSimPolicyWrapper._get_action; we cannot call the wrapper itself
    because it decodes actions to physical units, and candidates are scored in
    the model's normalized space.
    """
    out = {}
    for modality in ["video", "state", "language"]:
        out[modality] = {}
        for key in policy.modality_configs[modality].modality_keys:
            parsed = key if modality == "language" else f"{modality}.{key}"
            arr = flat[parsed]
            out[modality][key] = ([[str(x)] for x in arr] if modality == "language" else arr)
    return out

def _cpu_quota():
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            q, p = fh.read().split()
            if q != "max":
                return max(1, int(q) // int(p))
    except Exception:      # noqa: BLE001
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            q = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            p = int(fh.read())
        if q > 0:
            return max(1, q // p)
    except Exception:      # noqa: BLE001
        pass
    return os.cpu_count() or 4


_NPROC = _cpu_quota()
_POOL = ThreadPoolExecutor(max_workers=max(2, _NPROC - 2)) if _NPROC > 2 else None


def make_inputs(policy, obs_batch, actions=None):
    """Raw observations (+ optional expert actions) -> collated model inputs.

    Actions ride along inside VLAStepData so the processor normalises them with
    the same statistics the model was trained under; there is no standalone
    encoder to call.
    """
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import _rec_to_dtype

    def _one(o):
        flat = {
            "video.image": o["image"][None, None],
            "video.wrist_image": o["wrist_image"][None, None],
        }
        for i, k in enumerate(STATE_KEYS[:-1]):
            flat[k] = np.asarray([[o["state"][i]]], dtype=np.float32)[..., None]
        flat["state.gripper"] = np.asarray(o["state"][6:8], dtype=np.float32)[None, None]
        flat["annotation.human.action.task_description"] = [o["task"]]
        step = policy._to_vla_step_data(
            policy._unbatch_observation(flat_to_nested(policy, flat))[0])
        return step

    # The per-sample processor call (two 256x256 camera streams each) is pure
    # CPU and used to run single-threaded in a Python loop, which left the GPU
    # ~90% idle: 0.74 s/step against ~0.05-0.1 s of actual compute. The steps are
    # independent, so build them in a thread pool sized to the cgroup grant.
    steps = list(_POOL.map(_one, obs_batch)) if _POOL else [_one(o) for o in obs_batch]
    if actions is not None:
        akeys = policy.modality_configs["action"].modality_keys
        for i_s, step in enumerate(steps):
            a = np.asarray(actions[i_s], dtype=np.float32)  # (T, 7)
            step.actions = {k: a[:, i:i + 1] for i, k in enumerate(akeys)}
    processed = [policy.processor([{"type": MessageType.EPISODE_STEP.value,
                                    "content": st}]) for st in steps]
    collated = policy.collate_fn(processed)
    # collate_fn returns the kwargs get_action(**collated) takes, i.e. the real
    # payload sits under "inputs"; prepare_input wants that payload directly.
    collated = collated.get("inputs", collated)
    return _rec_to_dtype(collated, dtype=torch.bfloat16)


def flow_velocity(head, backbone_output, action_input, x_t, t):
    """Predicted velocity at a GIVEN (x_t, t) -- forward() samples its own.

    process_backbone_output writes its result back into the BatchFeature it is
    handed (vlln + vl_self_attention), so calling it twice on one object feeds
    already-normalised features through the stack a second time (measured at
    max|v1-v2| = 2.67). Each call therefore gets its own shallow copy: the
    tensors are shared but the write-back lands on a private dict.
    """
    backbone_output = head.process_backbone_output(
        type(backbone_output)({k: v for k, v in backbone_output.items()}))
    vl_embeds = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id
    state = action_input.state
    state = state.view(state.shape[0], 1, -1)
    state_features = head.state_encoder(state, embodiment_id)

    t_discretized = (t[:, 0, 0] * head.num_timestep_buckets).long()
    action_features = head.action_encoder(x_t, t_discretized, embodiment_id)
    # Official forward() adds this before concatenating; omitting it strips the
    # action tokens of any notion of position in the chunk, so the DiT cannot
    # tell step 0 from step 39. The velocity still points roughly the right way
    # (rectified-flow velocity is t-independent) which is why it showed up as a
    # corr of 0.53 rather than an outright failure.
    if head.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long,
                               device=action_features.device)
        action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
    sa_embs = torch.cat((state_features, action_features), dim=1)
    vl_attn_mask = backbone_output.backbone_attention_mask

    kw = dict(hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
              encoder_attention_mask=vl_attn_mask, timestep=t_discretized,
              return_all_hidden_states=True)
    if head.config.use_alternate_vl_dit:
        kw.update(image_mask=backbone_output.image_mask,
                  backbone_attention_mask=backbone_output.backbone_attention_mask)
    model_output, _ = head.model(**kw)
    pred = head.action_decoder(model_output, embodiment_id)
    return pred[:, -x_t.shape[1]:]
