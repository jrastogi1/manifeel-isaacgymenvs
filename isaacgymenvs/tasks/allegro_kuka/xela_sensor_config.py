# Copyright (c) 2024, NVIDIA Corporation
# All rights reserved.
#
# XELA uSkin tactile-sensor configuration for the 16-DOF Allegro hand.
#
# This module mirrors the sensor layout used by the Tactile-JEPA reference
# codebase (https://huggingface.co/Tactile-JEPA/tactile-jepa-xela) so that the
# simulated taxel observations match that setup:
#   - 18 XELA uSkin sensor patches mounted on the 4 fingers + palm.
#   - 368 taxels ("sensing points") in total, each reporting 3-axis touch
#     (X = shear, Y = shear, Z = normal).
#   - A strict, ordered flatten layout (XELA_FLATTEN_ORDER) that fixes the
#     column order of the (368, 3) taxel tensor.
#   - A hand-part grouping (thumb / index / middle / ring / palm) and a physical
#     connectivity graph (bridge link pairs) used by topology-aware encoders.
#
# Patch layout (from XELA "For Allegro Hand V4 Curved" product + Tactile-JEPA
# graph/utils.py):
#   - 4 curved fingertip patches (aftc):        30 taxels each  -> 120
#   - 11 flat phalange patches (4x4):           16 taxels each  -> 176
#   - 3 palm patches (4x6):                     24 taxels each  ->  72
#                                                              ----
#                                                        total  368
#
# Note on frames: each taxel reading is a 3-axis force. We obtain, per patch,
# the net contact force from PhysX, rotate it into the patch-local frame, and
# then spatially distribute it across that patch's taxels using each taxel's
# position on the sensor grid and the effective contact point implied by the
# patch wrench. See allegro_kuka_xela.py for the distribution scheme.

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Ordered sensor-patch layout (single source of truth for taxel ordering).
# Keys are the URDF link names of the XELA patch bodies; values are the number
# of taxels on that patch. The insertion order defines the flatten order of the
# (368, 3) observation, matching Tactile-JEPA's XELA_FLATTEN_ORDER exactly.
# ---------------------------------------------------------------------------
XELA_FLATTEN_ORDER: "OrderedDict[str, int]" = OrderedDict(
    [
        # thumb
        ("3aftc_palm_link", 30),
        ("link_15_4x4_palm_link", 16),
        ("link_14_4x4_palm_link", 16),
        # index finger
        ("0aftc_palm_link", 30),
        ("link_2_4x4_palm_link", 16),
        ("link_1A_4x4_palm_link", 16),
        ("link_1B_4x4_palm_link", 16),
        # middle finger
        ("1aftc_palm_link", 30),
        ("link_6_4x4_palm_link", 16),
        ("link_5A_4x4_palm_link", 16),
        ("link_5B_4x4_palm_link", 16),
        # ring finger
        ("2aftc_palm_link", 30),
        ("link_10_4x4_palm_link", 16),
        ("link_9A_4x4_palm_link", 16),
        ("link_9B_4x4_palm_link", 16),
        # palm
        ("ahr_palm_2_4x6_palm_link", 24),
        ("ahr_palm_1_4x6_palm_link", 24),
        ("ahr_palm_3_4x6_palm_link", 24),
    ]
)

# Which Allegro link each XELA patch body is rigidly attached to (its parent in
# the URDF). Fingertip patches attach to the distal phalanx (link_3); phalange
# patches attach to the proximal/medial phalanges; palm patches to the palm.
XELA_PATCH_PARENT_LINK: Dict[str, str] = {
    # thumb
    "3aftc_palm_link": "thumb_link_3",
    "link_15_4x4_palm_link": "thumb_link_1",
    "link_14_4x4_palm_link": "thumb_link_2",
    # index finger
    "0aftc_palm_link": "index_link_3",
    "link_2_4x4_palm_link": "index_link_2",
    "link_1A_4x4_palm_link": "index_link_1",
    "link_1B_4x4_palm_link": "index_link_0",
    # middle finger
    "1aftc_palm_link": "middle_link_3",
    "link_6_4x4_palm_link": "middle_link_2",
    "link_5A_4x4_palm_link": "middle_link_1",
    "link_5B_4x4_palm_link": "middle_link_0",
    # ring finger
    "2aftc_palm_link": "ring_link_3",
    "link_10_4x4_palm_link": "ring_link_2",
    "link_9A_4x4_palm_link": "ring_link_1",
    "link_9B_4x4_palm_link": "ring_link_0",
    # palm
    "ahr_palm_2_4x6_palm_link": "palm_link",
    "ahr_palm_1_4x6_palm_link": "palm_link",
    "ahr_palm_3_4x6_palm_link": "palm_link",
}

# Map each patch link to a coarse hand part (used for grouping / graph nodes).
LINK_TO_HAND_PART: Dict[str, str] = {
    "3aftc_palm_link": "thumb",
    "link_15_4x4_palm_link": "thumb",
    "link_14_4x4_palm_link": "thumb",
    "0aftc_palm_link": "index_finger",
    "link_2_4x4_palm_link": "index_finger",
    "link_1A_4x4_palm_link": "index_finger",
    "link_1B_4x4_palm_link": "index_finger",
    "1aftc_palm_link": "middle_finger",
    "link_6_4x4_palm_link": "middle_finger",
    "link_5A_4x4_palm_link": "middle_finger",
    "link_5B_4x4_palm_link": "middle_finger",
    "2aftc_palm_link": "ring_finger",
    "link_10_4x4_palm_link": "ring_finger",
    "link_9A_4x4_palm_link": "ring_finger",
    "link_9B_4x4_palm_link": "ring_finger",
    "ahr_palm_2_4x6_palm_link": "palm",
    "ahr_palm_1_4x6_palm_link": "palm",
    "ahr_palm_3_4x6_palm_link": "palm",
}

HAND_PART_ORDER: Tuple[str, ...] = (
    "thumb",
    "index_finger",
    "middle_finger",
    "ring_finger",
    "palm",
)

# Physical adjacency between neighbouring patch bodies (kinematic chain along
# each finger + palm bridges). Mirrors Tactile-JEPA's PHYSICAL_BRIDGE_LINK_PAIRS
# and is used to build the sensor connectivity graph for topology-aware masking.
PHYSICAL_BRIDGE_LINK_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("3aftc_palm_link", "link_15_4x4_palm_link"),
    ("link_15_4x4_palm_link", "link_14_4x4_palm_link"),
    ("link_14_4x4_palm_link", "ahr_palm_1_4x6_palm_link"),
    ("0aftc_palm_link", "link_2_4x4_palm_link"),
    ("link_2_4x4_palm_link", "link_1A_4x4_palm_link"),
    ("link_1A_4x4_palm_link", "link_1B_4x4_palm_link"),
    ("link_1B_4x4_palm_link", "ahr_palm_2_4x6_palm_link"),
    ("1aftc_palm_link", "link_6_4x4_palm_link"),
    ("link_6_4x4_palm_link", "link_5A_4x4_palm_link"),
    ("link_5A_4x4_palm_link", "link_5B_4x4_palm_link"),
    ("link_5B_4x4_palm_link", "ahr_palm_2_4x6_palm_link"),
    ("2aftc_palm_link", "link_10_4x4_palm_link"),
    ("link_10_4x4_palm_link", "link_9A_4x4_palm_link"),
    ("link_9A_4x4_palm_link", "link_9B_4x4_palm_link"),
    ("link_9B_4x4_palm_link", "ahr_palm_1_4x6_palm_link"),
    ("ahr_palm_1_4x6_palm_link", "ahr_palm_2_4x6_palm_link"),
    ("ahr_palm_2_4x6_palm_link", "ahr_palm_3_4x6_palm_link"),
)

# Total number of taxels (sensing points) across all patches.
NUM_XELA_TAXELS: int = int(sum(XELA_FLATTEN_ORDER.values()))
NUM_XELA_SENSORS: int = len(XELA_FLATTEN_ORDER)
# Each taxel reports 3-axis touch (X, Y, Z).
TAXEL_DIM: int = 3


def _build_patch_slices() -> "OrderedDict[str, Tuple[int, int]]":
    """Return, per patch link, the [start, end) slice into the flattened taxels."""
    slices: "OrderedDict[str, Tuple[int, int]]" = OrderedDict()
    start = 0
    for link_name, count in XELA_FLATTEN_ORDER.items():
        slices[link_name] = (start, start + count)
        start += count
    return slices


# Per-patch [start, end) offsets into the (368, 3) flattened taxel tensor.
XELA_PATCH_SLICES: "OrderedDict[str, Tuple[int, int]]" = _build_patch_slices()


def build_sensor_to_hand_part() -> Dict[int, str]:
    """Map every taxel id (0..367) to its coarse hand part."""
    sensor_to_part: Dict[int, str] = {}
    for link_name, count in XELA_FLATTEN_ORDER.items():
        start, end = XELA_PATCH_SLICES[link_name]
        part = LINK_TO_HAND_PART[link_name]
        for sensor_id in range(start, end):
            sensor_to_part[sensor_id] = part

    expected = set(range(NUM_XELA_TAXELS))
    actual = set(sensor_to_part.keys())
    if expected != actual:
        raise ValueError(
            f"XELA taxel mapping incomplete: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return sensor_to_part


# Per-taxel hand-part id in flatten order (values index into HAND_PART_ORDER).
SENSOR_TO_HAND_PART: Dict[int, str] = build_sensor_to_hand_part()


def get_sensor_grid(patch_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return the local (x, y) grid of taxel centers on a patch and standoff d.

    Replicates Tactile-JEPA's tactile_ssl/data/xela/utils.get_sensor_grid so the
    spatial layout of taxels matches the reference setup. The grid is expressed
    in the patch-local frame with the sensor surface normal along +Z; `d` is the
    surface stand-off (sensor thickness) along Z.

    Patch families:
      - "aftc"  : curved fingertip, 6x6 grid with corners removed -> 30 taxels.
      - "4x4"   : flat phalange, 4x4 grid                        -> 16 taxels.
      - "4x6"   : flat palm, 6x4 grid                            -> 24 taxels.
    """
    if "aftc" in patch_name:
        h, w, d = 0.031, 0.039, 0.029  # from mesh bounding box
        h_res, w_res = 6, 6
        x = np.linspace(0.5 - h_res / 2, h_res / 2 + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx_, yy_ = np.meshgrid(x, y)
        xx = np.concatenate([xx_[:4, :].flatten(), xx_[-2, 1:-1], xx_[-1, 2:-2]], axis=0)
        yy = np.concatenate([yy_[:4, :].flatten(), yy_[-2, 1:-1], yy_[-1, 2:-2]], axis=0)
    elif "4x4" in patch_name:
        h, w, d = 0.026, 0.024, 0.0044  # from mesh bounding box
        h_res, w_res = 4, 4
        x = np.linspace(0.5, h_res + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx, yy = np.meshgrid(x, y)
    elif "4x6" in patch_name:
        h, w, d = 0.052, 0.032, 0.0  # from mesh bounding box
        h_res, w_res = 6, 4
        x = np.linspace(0.5, h_res + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx, yy = np.meshgrid(x, y)
    else:
        raise ValueError(f"Unknown XELA patch type for {patch_name!r}")

    xx = np.asarray(xx).flatten()
    yy = np.asarray(yy).flatten()
    count = XELA_FLATTEN_ORDER[patch_name]
    if xx.shape[0] != count:
        raise ValueError(
            f"Grid for {patch_name!r} produced {xx.shape[0]} taxels, expected {count}"
        )
    return xx, yy, d


def patch_taxel_local_positions(patch_name: str) -> np.ndarray:
    """Return (num_taxels, 3) local taxel centers for a patch.

    Columns are (x, y, z) in the patch-local frame. The sensor surface lies in
    the local XY plane; taxels sit at the surface stand-off `d` along +Z.
    """
    xx, yy, d = get_sensor_grid(patch_name)
    positions = np.stack([xx, yy, np.full_like(xx, d)], axis=-1)
    return positions.astype(np.float32)


def build_patch_adjacency() -> Dict[str, List[str]]:
    """Undirected adjacency list over patch bodies (from bridge link pairs)."""
    adjacency: Dict[str, List[str]] = {link: [] for link in XELA_FLATTEN_ORDER}
    for a, b in PHYSICAL_BRIDGE_LINK_PAIRS:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return adjacency


def build_connectivity_graph() -> Tuple[np.ndarray, np.ndarray]:
    """Build a dense taxel-level connectivity graph from the patch adjacency.

    Two taxels are connected iff their parent patches are adjacent. Returns
    (num_taxels, num_taxels) int8 adjacency plus per-taxel group ids (hand part
    index). Useful for topology-aware masking / graph encoders.
    """
    part_order = {p: i for i, p in enumerate(HAND_PART_ORDER)}
    patch_adj = build_patch_adjacency()

    adjacency = np.zeros((NUM_XELA_TAXELS, NUM_XELA_TAXELS), dtype=np.int8)
    group_ids = np.zeros((NUM_XELA_TAXELS,), dtype=np.int64)

    for name_i in XELA_FLATTEN_ORDER.keys():
        start_i, end_i = XELA_PATCH_SLICES[name_i]
        group_ids[start_i:end_i] = part_order[LINK_TO_HAND_PART[name_i]]
        for neighbor in patch_adj[name_i]:
            start_j, end_j = XELA_PATCH_SLICES[neighbor]
            adjacency[start_i:end_i, start_j:end_j] = 1

    return adjacency, group_ids

