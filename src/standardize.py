from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from .profiles import ChannelProfile, normalize_ch_name

def standardize_channels(
    x: np.ndarray,                 # (C_i, T)
    ch_names: List[str],           # len == C_i
    profile: ChannelProfile,
    *,
    strict: bool = False,          # if True: require at least 1 matched channel
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map (x, ch_names) onto a reference channel profile.
    Returns:
      x_ref:   (C_ref, T) float32
      mask:    (C_ref,) bool   True where channel is present
      chan_ids:(C_ref,) int32  index ids [0..C_ref-1] where present else -1
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (C,T), got shape={x.shape}")
    C_i, T = x.shape
    if len(ch_names) != C_i:
        raise ValueError(f"len(ch_names)={len(ch_names)} must match x.shape[0]={C_i}")

    # normalize input channel names + apply aliases
    normed = []
    for name in ch_names:
        n = normalize_ch_name(name)
        n = profile.aliases.get(n, n)
        normed.append(n)

    # Build reference mapping
    ref_index = profile.index
    C_ref = len(profile.channels)

    x_ref = np.zeros((C_ref, T), dtype=np.float32)
    mask = np.zeros((C_ref,), dtype=bool)
    chan_ids = -np.ones((C_ref,), dtype=np.int32)

    # If duplicates map to same ref channel, keep the first occurrence (stable)
    filled = set()
    for src_i, n in enumerate(normed):
        j = ref_index.get(n)
        if j is None or j in filled:
            continue
        x_ref[j] = x[src_i].astype(np.float32, copy=False)
        mask[j] = True
        chan_ids[j] = j
        filled.add(j)

    if strict and not mask.any():
        raise ValueError("No channels matched the reference profile (strict=True).")

    return x_ref, mask, chan_ids
