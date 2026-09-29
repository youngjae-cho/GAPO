#!/usr/bin/env python
"""
Select a subset of preference pairs from a per-sample trajectory npz file
(produced by the GAPO trajectory logger) for data-efficiency experiments.

Inputs
------
- trajectory_npz: a .npz file with at least the keys
    sample_indices: (N,) int64
    margins, reward_margins, simpo_losses, gapo_losses, chosen_logps,
    rejected_logps: (T, N) float64    (T = number of checkpoints)
  Optional:
    is_flipped: (N,) int — used by --score=clean_oracle
    grad_norms: (N,) or (T, N) float — used by --score=low_gradnorm

Outputs
-------
A JSON file of the form
    {"score": "...", "selection": "stable|unstable|random|...",
     "fraction": 0.30, "indices": [int, int, ...], "size": K, "pool_size": N}

Score functions
---------------
- anchor_gap         Anchor Gap Gamma = (m_clean - m_perturbed) - gamma_beta_ratio,
                     i.e. the GAPO loss logit before sigmoid where the per-example
                     anchor margin is the SAM-perturbed margin from
                     `score_dataset.py`. Small Gamma => Stable.
- mean_reward_margin baseline: high reward margin => Stable
- final_reward_margin
- mean_simpo_loss    baseline: low loss => Stable
- mean_gapo_loss     baseline: low loss => Stable
- margin_variance    low variance => Stable (requires T>=2 ckpts)
- low_gradnorm       requires grad_norms in npz; low => Stable
- random
- clean_oracle       requires is_flipped; picks is_flipped==0 first

Selection modes
---------------
- stable        : keep top fraction by "stability" (per-score convention)
- unstable      : keep bottom fraction by stability (the high-Gamma tail)
- random        : uniform random
- remove_top    : keep ALL except top fraction by Gamma (high-Gamma removal)
- remove_random : keep ALL except a random fraction (control for remove_top)

The "anchor gap" Gamma convention used here:
- For each score, we define a per-sample stability s_i.
- Stable subset    = top-k by s_i (largest stability)
- Unstable subset  = bottom-k by s_i (smallest stability)
- For removal modes the ranking is by Gamma = -s_i (high Gamma = unstable).

Score-to-stability mapping
--------------------------
- mean_reward_margin / final_reward_margin: s = +margin (higher is more stable)
- mean_simpo_loss / mean_gapo_loss:         s = -loss   (lower loss is more stable)
- margin_variance:                          s = -var    (lower variance is more stable)
- low_gradnorm:                             s = -gradnorm
- random:                                   s = uniform random
- clean_oracle:                             s = -is_flipped + tiny noise tiebreaker
"""

import argparse
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np


def _resolve_gamma_beta_ratio(npz_path: str, override: Optional[float]) -> float:
    """Return gamma_beta_ratio. CLI override wins; otherwise read sidecar JSON;
    otherwise 0.0 (so Gamma reduces to plain margin, with a warning printed)."""
    if override is not None:
        return float(override)
    sidecar = npz_path + ".json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r") as f:
                meta = json.load(f)
            if "gamma_beta_ratio" in meta:
                return float(meta["gamma_beta_ratio"])
        except (OSError, ValueError, KeyError):
            pass
    print(
        "WARNING: gamma_beta_ratio not found in sidecar metadata and no --gamma_beta_ratio "
        "given. Defaulting to 0.0 (Gamma reduces to plain margin). Pass --gamma_beta_ratio "
        "to match the training config exactly.",
        file=sys.stderr,
    )
    return 0.0


def compute_stability(
    npz: np.lib.npyio.NpzFile,
    score: str,
    seed: int,
    gamma_beta_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sample_indices, stability) of equal length.

    Higher stability => more "Stable" sample.
    """
    sample_indices = npz["sample_indices"].astype(np.int64)
    rng = np.random.default_rng(seed)

    def mean_over_ckpts(name: str) -> np.ndarray:
        arr = np.asarray(npz[name])
        if arr.ndim == 2:
            return arr.mean(axis=0)
        return arr

    if score == "anchor_gap":
        # Anchor Gap Gamma_i = (m_clean_i - m_perturbed_i) - gamma_beta_ratio,
        # i.e. the GAPO loss logit (modulo a positive beta scaling that does
        # not affect ranking) where m = chosen_logp - rejected_logp at clean
        # weights and m_perturbed is at the SAM-perturbed weights.
        # Small Gamma => Stable, so stability = -Gamma.
        if "margins_perturbed" not in npz.files:
            raise ValueError(
                "anchor_gap requires 'margins_perturbed' in the trajectory npz; "
                "regenerate via score_dataset.py to record the SAM-perturbed margin."
            )
        m_clean = np.asarray(npz["margins"])
        m_pert = np.asarray(npz["margins_perturbed"])
        if m_clean.ndim == 2:
            m_clean = m_clean.mean(axis=0)
            m_pert = m_pert.mean(axis=0)
        gamma = (m_clean - m_pert) - gamma_beta_ratio
        stability = -gamma
    elif score == "mean_reward_margin":
        stability = mean_over_ckpts("reward_margins")
    elif score == "final_reward_margin":
        arr = np.asarray(npz["reward_margins"])
        stability = arr[-1] if arr.ndim == 2 else arr
    elif score == "mean_simpo_loss":
        stability = -mean_over_ckpts("simpo_losses")
    elif score == "mean_gapo_loss":
        stability = -mean_over_ckpts("gapo_losses")
    elif score == "margin_variance":
        arr = np.asarray(npz["reward_margins"])
        if arr.ndim != 2:
            raise ValueError("margin_variance requires (T,N) reward_margins (need >=2 checkpoints)")
        stability = -arr.var(axis=0)
    elif score == "low_gradnorm":
        if "grad_norms" not in npz.files:
            raise ValueError("low_gradnorm requires 'grad_norms' in the npz")
        arr = np.asarray(npz["grad_norms"])
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        stability = -arr
    elif score == "random":
        stability = rng.standard_normal(len(sample_indices))
    elif score == "clean_oracle":
        if "is_flipped" not in npz.files:
            raise ValueError("clean_oracle requires 'is_flipped' in the npz")
        flipped = np.asarray(npz["is_flipped"]).astype(np.float64)
        # noise tiebreak so we get a deterministic but uniform pick within the clean pool
        stability = -flipped + 1e-6 * rng.standard_normal(len(flipped))
    else:
        raise ValueError(f"Unknown score: {score}")

    return sample_indices, stability.astype(np.float64)


def select_indices(
    sample_indices: np.ndarray,
    stability: np.ndarray,
    fraction: float,
    mode: str,
    seed: int,
) -> np.ndarray:
    n = len(sample_indices)
    k = int(round(fraction * n))
    if k <= 0 or k > n:
        raise ValueError(f"fraction={fraction} yields k={k} (pool={n})")

    rng = np.random.default_rng(seed)

    if mode == "random":
        return np.sort(rng.choice(sample_indices, size=k, replace=False))

    # argsort ascending by stability: index 0 = least stable (high Gamma)
    order = np.argsort(stability, kind="stable")

    if mode == "stable":
        picked = order[-k:]
    elif mode == "unstable":
        picked = order[:k]
    elif mode == "remove_top":
        # remove top-k high-Gamma == bottom-k stability == order[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[order[:k]] = False
        picked = np.where(keep_mask)[0]
    elif mode == "remove_random":
        drop = rng.choice(n, size=k, replace=False)
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[drop] = False
        picked = np.where(keep_mask)[0]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return np.sort(sample_indices[picked])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory_npz", required=True, help="Path to all_checkpoints.npz")
    p.add_argument(
        "--score",
        default="anchor_gap",
        choices=[
            "anchor_gap",
            "mean_reward_margin",
            "final_reward_margin",
            "mean_simpo_loss",
            "mean_gapo_loss",
            "margin_variance",
            "low_gradnorm",
            "random",
            "clean_oracle",
        ],
    )
    p.add_argument(
        "--gamma_beta_ratio",
        type=float,
        default=None,
        help="gamma_beta_ratio used in the GAPO loss logit. If omitted, read from the "
             "trajectory's sidecar .json (written by score_dataset.py).",
    )
    p.add_argument(
        "--mode",
        default="stable",
        choices=["stable", "unstable", "random", "remove_top", "remove_random"],
    )
    p.add_argument("--fraction", type=float, required=True, help="Selection (or removal) fraction in (0, 1]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--report_flip_rate", action="store_true", help="If is_flipped is present, print flip rate of the selection")
    args = p.parse_args()

    if not (0.0 < args.fraction <= 1.0):
        sys.exit(f"--fraction must be in (0, 1]; got {args.fraction}")

    gamma_beta_ratio = _resolve_gamma_beta_ratio(args.trajectory_npz, args.gamma_beta_ratio)

    npz = np.load(args.trajectory_npz, allow_pickle=True)
    sample_indices, stability = compute_stability(
        npz, args.score, args.seed, gamma_beta_ratio=gamma_beta_ratio
    )
    picked = select_indices(sample_indices, stability, args.fraction, args.mode, args.seed)

    payload = {
        "score": args.score,
        "mode": args.mode,
        "fraction": args.fraction,
        "seed": args.seed,
        "gamma_beta_ratio": gamma_beta_ratio,
        "trajectory_npz": os.path.abspath(args.trajectory_npz),
        "pool_size": int(len(sample_indices)),
        "size": int(len(picked)),
        "indices": [int(x) for x in picked.tolist()],
    }

    if args.report_flip_rate and "is_flipped" in npz.files:
        flipped = np.asarray(npz["is_flipped"]).astype(bool)
        # map sample_idx -> flipped
        idx_to_flip = {int(s): bool(f) for s, f in zip(sample_indices.tolist(), flipped.tolist())}
        flipped_picked = sum(1 for i in picked.tolist() if idx_to_flip.get(int(i), False))
        payload["flip_rate"] = flipped_picked / max(1, len(picked))
        payload["flipped_in_pool"] = int(flipped.sum())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f)

    msg = (
        f"score={args.score} mode={args.mode} fraction={args.fraction} "
        f"pool={payload['pool_size']} -> selected={payload['size']}"
    )
    if "flip_rate" in payload:
        msg += f"  flip_rate={payload['flip_rate']:.4f}"
    print(msg)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
